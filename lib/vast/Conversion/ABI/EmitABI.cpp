// Copyright (c) 2021-present, Trail of Bits, Inc.

#include "vast/Conversion/Passes.hpp"

VAST_RELAX_WARNINGS
#include <mlir/Analysis/DataLayoutAnalysis.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Conversion/LLVMCommon/Pattern.h>

#include <mlir/Rewrite/PatternApplicator.h>

#include <llvm/ADT/APFloat.h>
VAST_UNRELAX_WARNINGS

#include "../PassesDetails.hpp"

#include "vast/Conversion/Common/Patterns.hpp"
#include "vast/Conversion/TypeConverters/TypeConverter.hpp"

#include "vast/Dialect/HighLevel/HighLevelAttributes.hpp"
#include "vast/Dialect/HighLevel/HighLevelTypes.hpp"
#include "vast/Dialect/HighLevel/HighLevelOps.hpp"

#include "vast/Dialect/LowLevel/LowLevelOps.hpp"

#include "vast/ABI/ABI.hpp"
#include "vast/ABI/Driver.hpp"

#include "vast/Util/Common.hpp"
#include "vast/Util/Functions.hpp"
#include "vast/Util/DialectConversion.hpp"
#include "vast/Util/Symbols.hpp"

#include "vast/Dialect/ABI/ABIOps.hpp"

#include <gap/core/overloads.hpp>

#include <iostream>
#include <unordered_map>

namespace vast
{
    template< typename FnOp >
    std::vector< mlir::Location > collect_arg_locs(FnOp op)
    {
        VAST_ASSERT(!op.getBody().empty());
        std::vector< mlir::Location > out;
        for (auto arg : op.getBody().front().getArguments())
            out.push_back(arg.getLoc());
        return out;
    }

    template< typename Op >
    using abi_info_map_t = std::unordered_map< std::string, abi::func_info< Op > >;

    template< typename R, typename RootOp, typename DL >
    auto collect_abi_info(RootOp root_op, const DL &dl)
        -> abi_info_map_t< R >
    {
        abi_info_map_t< R > out;
        auto gather = [&](R op, const mlir::WalkStage &)
        {
            auto name = op.getName();
            out.emplace( name.str(), abi::make_x86_64(op, dl) );

            return mlir::WalkResult::advance();
        };

        root_op->walk(gather);
        return out;
    }

    // TODO(conv:abi): Remove as we most likely do not need this.
    struct TypeConverter : conv::tc::mixins< TypeConverter >,
                           conv::tc::identity_type_converter
    {
        TypeConverter(const mlir::DataLayout &dl, mcontext_t &mctx)
            : conv::tc::identity_type_converter(), dl(dl), mctx(mctx)
        {}

        const mlir::DataLayout &dl;
        mcontext_t &mctx;
    };

    namespace
    {
        template< typename Self >
        struct abi_info_utils
        {
            using types_t = std::vector< mlir::Type >;
            using abi_info_t = abi::func_info< hl::FuncOp >;

            const auto &self() const { return static_cast< const Self & >(*this); }
            auto &self() { return static_cast< Self & >(*this); }

            auto get_inserter(auto &into)
            {
                return gap::overloaded {
                    [ & ](auto vals) { into.insert(into.end(), vals.begin(), vals.end()); },
                    [ & ](mlir_value val) { into.push_back(val); }
                };
            };

            bool returns_value()
            {
                // TODO(conv:abi): `sret`?
                for (const auto &e : self().abi_info.rets())
                    if (!std::holds_alternative< abi::ignore >(e.style))
                        return true;
                return false;
            }

            types_t abified_args() const
            {
                types_t out;
                for (const auto &e : self().abi_info.args())
                {
                    auto trgs = e.target_types();
                    out.insert(out.end(), trgs.begin(), trgs.end());
                }
                return out;
            }

            types_t abified_rets()
            {
                types_t out;
                for (const auto &e : self().abi_info.rets())
                {
                    auto trgs = e.target_types();
                    out.insert(out.end(), trgs.begin(), trgs.end());
                }
                return out;
            }

            core::FunctionType abified_type()
            {
                auto add_lvalue = [&](const auto &from)
                {
                    types_t out;
                    for (auto t : from)
                        out.push_back(hl::LValueType::get(t.getContext(), t));
                    return out;
                };

                return  core::FunctionType::get(add_lvalue(abified_args()),
                                                abified_rets());
            }

            // TODO(conv:abi): Can be replaced with `llvm::zip`? Will work with
            //                 all other classifications?
            void zip(const auto &a, const auto &b, auto &&yield)
            {
                auto a_it = a.begin();
                auto b_it = b.begin();
                while (a_it != a.end() && b_it != b.end())
                {
                    yield(*a_it, *b_it);
                    ++a_it; ++b_it;
                }
                VAST_ASSERT(a_it == a.end() && b_it == b.end());
            }

            void zip_ret(const auto &abi, const auto &results, const auto &concretes,
                         auto &&yield)
            {
                // These should be always synced, but we are defensive as this may
                // be called with some random module that is not a result of our
                // classification.
                VAST_ASSERT(std::ranges::size(abi) == std::ranges::size(results));
                auto abi_it = abi.begin();
                auto result_it = results.begin();

                auto advance = [&]() { ++abi_it; ++result_it; };

                auto args_it = concretes.begin();

                while (abi_it != abi.end() && args_it != concretes.end())
                {
                    std::vector< mlir::Value > collect;
                    for (std::size_t i = 0; i < abi_it->target_types().size(); ++i)
                    {
                        VAST_ASSERT(args_it != concretes.end());
                        collect.push_back(*(args_it++));
                    }
                    yield(*abi_it, *result_it, std::move(collect));

                    advance();
                }
            }

        };

        template< typename op_t >
        struct abi_transform : match_and_rewrite_state_capture< op_t >,
                               abi_info_utils< abi_transform< op_t > >
        {
            using state_t = match_and_rewrite_state_capture< op_t >;
            using abi_utils = abi_info_utils< abi_transform< op_t > >;
            using abi_info_t = typename abi_utils::abi_info_t;

            using state_t::op;
            using state_t::operands;
            using state_t::rewriter;

            const abi_info_t &abi_info;

            using materialized_args_t = std::vector< mlir::Value >;
            using mapped_arg_t = std::tuple< mlir::Type, materialized_args_t >;

            abi_transform(state_t state, const abi_info_t &abi_info)
                : state_t(std::move(state)), abi_info(abi_info)
            {}

            using types_t = std::vector< mlir::Type >;

            abi::FuncOp make()
            {
                mlir::SmallVector< mlir::DictionaryAttr, 8 > arg_attrs;
                mlir::SmallVector< mlir::NamedAttribute, 8 > other_attrs;

                op.getAllArgAttrs(arg_attrs);
                auto wrapper = rewriter.template create< abi::FuncOp >(
                        op.getLoc(),
                        // Temporal, to avoid verification issues, will be changed once
                        // original func is removed.
                        "vast.abi" + op.getName().str(),
                        this->abified_type(),
                        core::GlobalLinkageKind::InternalLinkage,
                        other_attrs,
                        arg_attrs
                );

                // Copying visibility from the original function results in error?
                wrapper.setVisibility(mlir::SymbolTable::Visibility::Private);

                mk_prologue(wrapper);

                return wrapper;
            }

            auto mk_direct(auto &bld, auto loc, const abi::direct &abi_arg,
                           const mapped_arg_t &entry)
            {
                auto &[original_type, args] = entry;

                auto opaque = rewriter.template create< abi::DirectOp >(
                        op.getLoc(),
                        original_type,
                        args);
                return opaque.getResults();
            }

            auto fold_arg(auto &bld, auto loc,
                          const abi::direct &abi_arg, const mapped_arg_t &entry)
                -> mlir::ResultRange
            {
                return mk_direct(bld, loc, abi_arg, entry);
            }

            auto fold_arg(auto &bld, auto loc,
                          const abi::indirect &abi_arg, const mapped_arg_t &entry)
                -> mlir_value
            {
                auto &[original_type, args] = entry;
                return bld.template create< abi::IndirectOp >(
                    op.getLoc(),
                    original_type,
                    args);
            }

            auto fold_arg(auto &, auto, const auto &abi_arg, const mapped_arg_t &)
                -> mlir::ResultRange
            {
                VAST_TODO("conv:abi:fold_arg unsupported arg_info: {0}", abi_arg.to_string());
            }

            auto fold_arg(auto &, auto, const std::monostate &, const mapped_arg_t &)
                -> mlir::ResultRange
            {
                VAST_TODO("conv:abi:fold_arg unsupported arg_info: monostate");
            }

            auto fold_ret(auto &bld, auto loc,
                          const abi::direct &abi_arg, const mapped_arg_t &entry)
                -> mlir::ResultRange
            {
                return mk_direct(bld, loc, abi_arg, entry);
            }

            auto fold_ret(auto &, auto, const auto &abi_arg, const mapped_arg_t &)
                -> mlir::ResultRange
            {
                VAST_TODO("conv:abi:fold_ret unsupported arg_info: {0}", abi_arg.to_string());
            }

            auto fold_ret(auto &, auto, const std::monostate &, const mapped_arg_t &)
                -> mlir::ResultRange
            {
                VAST_TODO("conv:abi:fold_ret unsupported arg_info: monostate");
            }

            void mk_prologue(abi::FuncOp func)
            {
                std::vector< mlir::Location > arg_locs;
                auto process = [&](const auto &arg_info, auto loc)
                {
                    for (std::size_t i = 0; i < arg_info.target_types().size(); ++i)
                        arg_locs.push_back(loc);
                };
                // TODO(conv:abi): This won't work with `sret`.
                this->zip(abi_info.args(), collect_arg_locs(op), process);


                auto guard = mlir::OpBuilder::InsertionGuard(rewriter);
                auto entry = rewriter.createBlock(&func.getBody(), {},
                                                  func.getFunctionType().getInputs(),
                                                  arg_locs);

                // Compute some mappings that will be needed later.
                // arg_info -> { original type, [ arguments in the new function] }
                std::unordered_map< const abi::arg_info *, mapped_arg_t > arg_to_locals;

                auto func_arg_it = func.args_begin();
                auto op_arg_it = op.args_begin();
                for (std::size_t i = 0; i < abi_info.args().size(); ++i)
                {
                    auto &current = abi_info.args()[i];

                    materialized_args_t mat_args;
                    for (std::size_t j = 0; j < current.target_types().size(); ++j)
                        mat_args.push_back( *(func_arg_it++) );

                    auto entry = std::make_tuple((op_arg_it++)->getType(), std::move(mat_args));
                    arg_to_locals[ &current ] = std::move(entry);
                }

                rewriter.setInsertionPointToStart(entry);

                materialized_args_t to_yield;
                auto fold_all_args = [&](auto &bld, auto loc)
                {
                    auto store = this->get_inserter(to_yield);

                    for (const auto &abi_arg : abi_info.args())
                    {
                        VAST_ASSERT(arg_to_locals.count(&abi_arg));
                        const auto &entry = arg_to_locals[ &abi_arg ];

                        auto dispatch = [&](const auto &arg)
                        {
                            return store(fold_arg(bld, loc, arg, entry));
                        };
                        std::visit(dispatch, abi_arg.style);
                    }

                    bld.template create< abi::YieldOp >(
                            loc, op.getFunctionType().getResults(), to_yield );
                };

                auto abi_prologue = rewriter.template create< abi::PrologueOp >(
                        op.getLoc(),
                        op.getFunctionType().getInputs(),
                        fold_all_args);

                get_body(func, abi_prologue.getResults());

                if (abi_prologue.getResults().empty())
                    rewriter.eraseOp(abi_prologue);
            }

            void get_body(abi::FuncOp func, const auto &arg_mapping)
            {

                rewriter.cloneRegionBefore(op.getBody(), func.getBody(),
                                            func.getBody().end());

                auto entry = &*func.getBody().begin();
                auto original_entry = &*(std::next(func.getBody().begin()));
                rewriter.mergeBlocks(original_entry, entry, arg_mapping);
            }

        };

        template< typename op_t >
        struct call_wrapper : abi_info_utils< call_wrapper< op_t > >,
                              match_and_rewrite_state_capture< op_t >
        {
            using state_t = match_and_rewrite_state_capture< op_t >;

            using abi_utils = abi_info_utils< call_wrapper< op_t > >;
            using abi_info_t = typename abi_utils::abi_info_t;

            using state_t::op;
            using state_t::operands;
            using state_t::rewriter;

            const abi_info_t &abi_info;

            call_wrapper(state_t state, const abi_info_t &abi_info)
                : state_t(std::move(state)), abi_info(abi_info)
            {}

            using values_t = std::vector< mlir::Value >;

            template< typename Impl >
            auto mk_direct(auto &bld, auto loc,
                           const abi::direct &arg, values_t concrete_args)
                -> values_t
            {
                auto vals = rewriter.template create< Impl >(
                        op.getLoc(),
                        arg.target_types,
                        concrete_args).getResults();
                return { vals.begin(), vals.end() };
            }

            auto mk_ret(auto &bld, auto loc,
                        const abi::direct &arg,
                        mlir::Value result,
                        values_t concrete_args)
                -> values_t
            {
                auto vals = rewriter.template create< abi::DirectOp >(
                        loc,
                        result.getType(),
                        concrete_args).getResults();
                return { vals.begin(), vals.end() };
            }

            auto mk_ret(auto &, auto loc,
                        const abi::ignore &,
                        mlir::Value,
                        values_t concrete_args)
                -> values_t
            {
                return concrete_args;
            }

            auto mk_ret(auto &, auto, const auto &abi_arg, mlir::Value, values_t)
                -> mlir::ResultRange
            {
                VAST_TODO("conv:abi:mk_ret unsupported arg_info: {0}", abi_arg.to_string());
            }

            auto mk_ret(auto &, auto, const std::monostate &, mlir::Value, values_t)
                -> mlir::ResultRange
            {
                VAST_TODO("conv:abi:mk_ret unsupported arg_info: monostate");
            }

            auto mk_arg(auto &bld, auto loc, const abi::direct &arg, mlir::Value concrete_arg)
            {
                return mk_direct< abi::DirectOp >(bld, loc, arg, { concrete_arg });
            }

            auto mk_arg(auto &bld, auto loc, const abi::indirect &arg, mlir::Value concrete_arg)
                -> mlir_value
            {
                auto vals = bld.template create< abi::IndirectOp >(
                    op.getLoc(),
                    arg.target_types,
                    concrete_arg).getResult();
                return vals;
            }

            auto mk_arg(auto &, auto, const auto &abi_arg, const mlir::Value &)
                -> mlir::ResultRange
            {
                VAST_TODO("conv:abi:mk_arg unsupported arg_info: {0}", abi_arg.to_string());
            }

            auto mk_arg(auto &, auto, const std::monostate &, const mlir::Value &)
                -> mlir::ResultRange
            {
                VAST_TODO("conv:abi:mk_arg unsupported arg_info: monostate");
            }

            auto execution_region_maker()
            {
                return [&](auto &bld, auto loc)
                {
                    auto args = [ & ]() -> values_t
                    {
                        if (this->abified_args().empty())
                            return {};

                        auto cooked_args = bld.template create< abi::CallArgsOp >(
                                loc,
                                this->abified_args(),
                                args_maker()).getResults();
                        return { cooked_args.begin(), cooked_args.end() };
                    }();

                    auto call = bld.template create< abi::CallOp >(
                            loc,
                            op.getCallee(),
                            this->abified_rets(),
                            args);

                    auto to_yield = [ & ]() -> mlir::ResultRange
                    {
                        if (!this->returns_value())
                            return call.getResults();

                        return bld.template create< abi::CallRetsOp >(
                                loc,
                                op.getResults().getType(),
                                rets_maker(call.getResults())).getResults();
                    }();

                    bld.template create< abi::YieldOp >(
                            loc,
                            op.getResults().getType(),
                            to_yield);
                };
            }

            auto args_maker()
            {
                return [&](auto &bld, auto loc)
                {
                    std::vector< mlir::Value > out;
                    auto store = this->get_inserter(out);

                    auto process = [&](const auto &arg_info, auto val)
                    {
                        auto dispatch = [&](const auto &abi_arg)
                        {
                            return store(mk_arg(bld, loc, abi_arg, val));
                        };
                        std::visit(dispatch, arg_info.style);
                    };

                    this->zip(abi_info.args(), op.getArgOperands(), process);

                    bld.template create< abi::YieldOp >(
                        loc,
                        this->abified_args(),
                        out);
                };

            }

            auto rets_maker(mlir::ValueRange vals)
            {
                return [=, this](auto &bld, auto loc)
                {
                    std::vector< mlir::Value > out;
                    auto store = [&](auto vals)
                    {
                        out.insert(out.end(), vals.begin(), vals.end());
                    };

                    auto process = [&](const auto &arg_info, auto result, auto vals)
                    {
                        auto dispatch = [&](const auto &abi_arg)
                        {
                            return store(mk_ret(bld, loc, abi_arg, result, vals));
                        };
                        std::visit(dispatch, arg_info.style);
                    };

                    this->zip_ret(abi_info.rets(), op.getResults(), vals, process);

                    bld.template create< abi::YieldOp >(
                        loc,
                        this->abified_rets(),
                        out);
                };
            }

            auto make()
            {
                return rewriter.template create< abi::CallExecutionOp >(
                        op.getLoc(),
                        op.getCallee(),
                        op.getResults().getType(),
                        op.getArgOperands(),
                        execution_region_maker());
            }
        };


        template< typename op_t >
        struct return_wrapper : abi_info_utils< return_wrapper< op_t > >,
                                match_and_rewrite_state_capture< op_t >
        {
            using state_t = match_and_rewrite_state_capture< op_t >;

            using abi_utils = abi_info_utils< return_wrapper< op_t > >;
            using abi_info_t = typename abi_utils::abi_info_t;

            using state_t::op;
            using state_t::operands;
            using state_t::rewriter;

            const abi_info_t &abi_info;

            return_wrapper(state_t state, const abi_info_t &abi_info)
                : state_t(std::move(state)), abi_info(abi_info)
            {}

            using values_t = std::vector< mlir::Value >;

            template< typename Impl >
            auto mk_direct(auto &bld, auto loc,
                           const abi::direct &arg, mlir::Value concrete_arg)
                -> values_t
            {
                auto vals = rewriter.template create< Impl >(
                        op.getLoc(),
                        arg.target_types,
                        concrete_arg).getResults();
                return { vals.begin(), vals.end() };
            }

            auto mk_ret(auto &bld, auto loc, const abi::direct &arg, mlir::Value concrete_arg)
            {
                return mk_direct< abi::DirectOp >(bld, loc, arg, concrete_arg);
            }

            auto mk_ret(auto &, auto, const auto &abi_arg, const mlir::Value &)
                -> mlir::ResultRange
            {
                VAST_TODO("conv:abi:mk_ret unsupported arg_info: {0}", abi_arg.to_string());
            }

            auto mk_ret(auto &, auto, const std::monostate &, const mlir::Value &)
                -> mlir::ResultRange
            {
                VAST_TODO("conv:abi:mk_ret unsupported arg_info: monostate");
            }

            auto wrap_return(mlir::ValueRange vals)
            {
                return [=, this](auto &bld, auto loc)
                {
                    std::vector< mlir::Value > out;
                    auto store = [&](auto to_store)
                    {
                        out.insert(out.end(), to_store.begin(), to_store.end());
                    };

                    auto process = [&](const auto &arg_info, auto val)
                    {
                        auto dispatch = [&](const auto &abi_arg)
                        {
                            return store(mk_ret(bld, loc, abi_arg, val));
                        };
                        std::visit(dispatch, arg_info.style);
                    };

                    this->zip(abi_info.rets(), vals, process);

                    bld.template create< abi::YieldOp >(
                        loc,
                        this->abified_rets(),
                        out);
                };
            }

            hl::ReturnOp make()
            {
                auto wrapped = rewriter.template create< abi::EpilogueOp >(
                        op.getLoc(),
                        this->abified_rets(),
                        wrap_return(op.getResult()));

                return rewriter.template create< hl::ReturnOp >(
                        op.getLoc(),
                        wrapped.getResults());
            }

        };

        template< typename Op >
        struct func_type : mlir::OpConversionPattern< Op >
        {
            using Base = mlir::OpConversionPattern< Op >;

            TypeConverter &tc;
            const abi_info_map_t< hl::FuncOp > &abi_info_map;

            func_type(TypeConverter &tc,
                      const abi_info_map_t< hl::FuncOp > &abi_info_map,
                      mcontext_t &mctx)
                : Base(tc, &mctx), tc(tc), abi_info_map(abi_info_map)
            {}

            mlir::LogicalResult matchAndRewrite(
                    Op op, typename Op::Adaptor ops,
                    conversion_rewriter &rewriter) const override
            {
                auto abi_map_it = abi_info_map.find(op.getName().str());
                if (abi_map_it == abi_info_map.end())
                    return mlir::failure();

                const auto &abi_info = abi_map_it->second;
                abi_transform< Op >({ op, ops, rewriter }, abi_info).make();
                rewriter.eraseOp(op);
                return mlir::success();
            }
        };

        struct call_op : mlir::OpConversionPattern< hl::CallOp >
        {
            using Base = mlir::OpConversionPattern< hl::CallOp >;
            using Op = hl::CallOp;

            TypeConverter &tc;
            const abi_info_map_t< hl::FuncOp > &abi_info_map;

            call_op(TypeConverter &tc, const abi_info_map_t< hl::FuncOp > &abi_info_map,
                    mcontext_t &mctx)
                : Base(tc, &mctx), tc(tc), abi_info_map(abi_info_map)
            {}

            mlir::LogicalResult matchAndRewrite(
                    Op op, typename Op::Adaptor ops,
                    conversion_rewriter &rewriter) const override
            {
                auto abi_map_it = abi_info_map.find(op.getCallee().str());
                if (abi_map_it == abi_info_map.end())
                    return mlir::failure();

                const auto &abi_info = abi_map_it->second;
                auto call = call_wrapper< Op >({op, ops, rewriter}, abi_info).make();
                rewriter.replaceOp(op, call);
                return mlir::success();
            }
        };

        struct return_op : mlir::OpConversionPattern< hl::ReturnOp >
        {
            using Base = mlir::OpConversionPattern< hl::ReturnOp >;
            using Op = hl::ReturnOp;

            TypeConverter &tc;
            const abi_info_map_t< hl::FuncOp > &abi_info_map;

            return_op(TypeConverter &tc,
                      const abi_info_map_t< hl::FuncOp > &abi_info_map,
                      mcontext_t &mctx)
                : Base(tc, &mctx), tc(tc), abi_info_map(abi_info_map)
            {}

            mlir::LogicalResult matchAndRewrite(
                    Op op, typename Op::Adaptor ops,
                    conversion_rewriter &rewriter) const override
            {
                auto func = op->getParentOfType< abi::FuncOp >();
                if (!func)
                    return mlir::failure();

                auto name = func.getName();
                if (!name.consume_front("vast.abi"))
                    return mlir::failure();

                auto abi_map_it = abi_info_map.find(name.str());
                if (abi_map_it == abi_info_map.end())
                    return mlir::failure();

                const auto &abi_info = abi_map_it->second;
                return_wrapper< Op >({op, ops, rewriter}, abi_info).make();

                rewriter.eraseOp(op);
                return mlir::success();
            }
        };

    } // namespace


    struct EmitABI : EmitABIBase< EmitABI >
    {
        using target_t = mlir::ConversionTarget;
        using patterns_t = mlir::RewritePatternSet;
        using phase_t = std::tuple< target_t, patterns_t >;

        phase_t first_phase(auto &tc, const auto &abi_info_map)
        {
            auto &mctx = this->getContext();

            mlir::ConversionTarget target(mctx);
            target.markUnknownOpDynamicallyLegal([](auto) { return true; });
            target.addIllegalOp< hl::CallOp >();

            mlir::RewritePatternSet patterns(&mctx);
            patterns.add< call_op >(tc, abi_info_map, mctx);

            return { std::move(target), std::move(patterns) };
        }

        phase_t second_phase(auto &tc, const auto &abi_info_map)
        {

            auto &mctx = this->getContext();

            mlir::ConversionTarget target(mctx);
            target.markUnknownOpDynamicallyLegal([](auto) { return true; });

            auto should_transform = [&](hl::FuncOp op)
            {
                // TODO(abi): Due to some issues with location info of arguments
                //            declaration are not yet supported.
                return op.getName() == "main" && !op.isDeclaration();
            };

            target.addDynamicallyLegalOp< hl::FuncOp >(should_transform);

            mlir::RewritePatternSet patterns(&mctx);
            patterns.add< func_type< hl::FuncOp > >(tc, abi_info_map, mctx);

            return { std::move(target), std::move(patterns) };
        }

        phase_t third_phase(auto &tc, const auto &abi_info_map)
        {
            auto &mctx = this->getContext();

            mlir::ConversionTarget target(mctx);
            target.markUnknownOpDynamicallyLegal([](auto) { return true; });

            // Plan is to still leave `hl.return` but it should return values
            // yielded by `abi.epilogue`.
            auto is_return_legal = [&](hl::ReturnOp op)
            {
                auto func = op->getParentOfType< abi::FuncOp >();
                if (!func || func.getName() == "main")
                    return true;

                for (auto val : op.getResult())
                    if (!val.getDefiningOp< abi::EpilogueOp >())
                        return false;
                return true;
            };

            target.addDynamicallyLegalOp< hl::ReturnOp >(is_return_legal);

            mlir::RewritePatternSet patterns(&mctx);
            patterns.add< return_op >(tc, abi_info_map, mctx);

            return { std::move(target), std::move(patterns) };
        }

        mlir::LogicalResult run(phase_t phase)
        {
            auto [trg, patterns] = std::move(phase);
            return mlir::applyPartialConversion(this->getOperation(), trg, std::move(patterns));
        }

        void runOnOperation() override
        {
            auto &mctx = this->getContext();
            mlir::ModuleOp op = this->getOperation();

            const auto &dl_analysis = this->getAnalysis< mlir::DataLayoutAnalysis >();
            auto tc = TypeConverter(dl_analysis.getAtOrAbove(op), mctx);
            auto abi_info_map = collect_abi_info< hl::FuncOp >(
                    op, dl_analysis.getAtOrAbove(op));

            if (mlir::failed(run(first_phase(tc, abi_info_map))))
                return signalPassFailure();

            if (mlir::failed(run(second_phase(tc, abi_info_map))))
                return signalPassFailure();

            if (mlir::failed(run(third_phase(tc, abi_info_map))))
                return signalPassFailure();
        }
    };

} // namespace vast

std::unique_ptr< mlir::Pass > vast::createEmitABIPass()
{
    return std::make_unique< vast::EmitABI >();
}
