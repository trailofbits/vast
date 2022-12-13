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

#include "vast/Dialect/HighLevel/HighLevelAttributes.hpp"
#include "vast/Dialect/HighLevel/HighLevelTypes.hpp"
#include "vast/Dialect/HighLevel/HighLevelOps.hpp"

#include "vast/Dialect/LowLevel/LowLevelOps.hpp"

#include "vast/ABI/ABI.hpp"
#include "vast/ABI/Driver.hpp"

#include "vast/Util/Common.hpp"
#include "vast/Util/Functions.hpp"
#include "vast/Util/TypeConverter.hpp"
#include "vast/Util/DialectConversion.hpp"
#include "vast/Util/LLVMTypeConverter.hpp"
#include "vast/Util/Symbols.hpp"

#include "vast/Dialect/ABI/ABIOps.hpp"

#include <iostream>

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

    struct TypeConverter : util::TCHelpers< TypeConverter >, mlir::TypeConverter
    {
        TypeConverter(const mlir::DataLayout &dl, MContext &mctx)
            : dl(dl), mctx(mctx)
        {}

        const mlir::DataLayout &dl;
        MContext &mctx;
    };

    namespace pattern
    {
        template< typename Op >
        struct wrapper_builder
        {
            using self_t = wrapper_builder< Op >;
            using op_t = Op;
            using operands_t = typename op_t::Adaptor;
            using func_info_t = abi::func_info< op_t >;

            op_t op;
            operands_t operands;
            mlir::ConversionPatternRewriter &rewriter;
            abi::WrapFuncOp wrapper;
            func_info_t func_info;

            wrapper_builder(Op op, operands_t operands,
                            mlir::ConversionPatternRewriter &rewriter,
                            abi::WrapFuncOp wrapper,
                            func_info_t func_info)
                : op(op), operands(operands), rewriter(rewriter), wrapper(wrapper),
                  func_info(std::move(func_info))
            {}

            using values_t = std::vector< mlir::Value >;

            auto direct(const abi::direct &arg, mlir::Value concrete_arg)
                -> values_t
            {
                auto vals = rewriter.create< abi::DirectOp >(
                        op.getLoc(),
                        arg.target_types,
                        concrete_arg).getResults();
                return { vals.begin(), vals.end() };
            }

            template< typename Impl >
            auto mk_direct(const abi::direct &arg, mlir::Value concrete_arg)
                -> values_t
            {
                auto vals = rewriter.create< Impl >(
                        op.getLoc(),
                        arg.target_types,
                        concrete_arg).getResults();
                return { vals.begin(), vals.end() };
            }

            auto mk_ret(const abi::direct &arg, mlir::Value concrete_arg)
            {
                return mk_direct< abi::RetDirectOp >(arg, concrete_arg);
            }

            auto mk_arg(const abi::direct &arg, mlir::Value concrete_arg)
            {
                return mk_direct< abi::DirectOp >(arg, concrete_arg);
            }

            auto unknown()
            {
                VAST_TODO("conv:abi: Classification returned std::monostate!");
            }

            auto unsupported(const auto &info, mlir::Value concrete_arg)
                -> values_t
            {
                VAST_TODO("conv:abi: unsupported arg_info: {0}", info.to_string());
                return {};
            }

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

            std::vector< mlir::Value > emit_prologue()
            {
                VAST_ASSERT(func_info.args().size() == wrapper.getArguments().size());

                std::vector< mlir::Value > out;
                auto store = [&](auto vals)
                {
                    out.insert(out.end(), vals.begin(), vals.end());
                };

                auto process = [&](const auto &arg_info, auto val)
                {
                    std::visit(overloaded {
                        [&](const abi::direct &arg) { return store(mk_arg(arg, val)); },
                        [&](const abi::extend &arg) { return store(unsupported(arg, val)); },
                        [&](const abi::indirect &arg) { return store(unsupported(arg, val)); },
                        [&](const abi::ignore &arg) { return store(unsupported(arg, val)); },
                        [&](const abi::expand &arg) { return store(unsupported(arg, val)); },
                        [&](const abi::coerce_and_expand &arg)
                        {
                            return store(unsupported(arg, val));
                        },
                        [&](const abi::in_alloca &arg) { return store(unsupported(arg, val)); },
                        [&](const std::monostate &arg) { return unknown(); },
                    }, arg_info.style);
                };

                zip(func_info.args(), wrapper.getArguments(), process);


                return out;
            }

            std::vector< mlir::Value > emit_call( values_t args )
            {
                auto val = rewriter.create< abi::CallOp >(
                        wrapper.getLoc(),
                        op.getName(),
                        op.getFunctionType().getResults(),
                        args ).getResult();

                return { val };
            }

            auto emit_epilogue( values_t rets )
            {
                std::vector< mlir::Value > out;
                auto store = [&](auto vals)
                {
                    out.insert(out.end(), vals.begin(), vals.end());
                };

                auto process = [&](const auto &arg_info, auto val)
                {
                    std::visit(overloaded {
                        [&](const abi::direct &arg) { return store(mk_ret(arg, val)); },
                        [&](const abi::extend &arg) { return store(unsupported(arg, val)); },
                        [&](const abi::indirect &arg) { return store(unsupported(arg, val)); },
                        [&](const abi::ignore &arg) { return store(unsupported(arg, val)); },
                        [&](const abi::expand &arg) { return store(unsupported(arg, val)); },
                        [&](const abi::coerce_and_expand &arg)
                        {
                            return store(unsupported(arg, val));
                        },
                        [&](const abi::in_alloca &arg) { return store(unsupported(arg, val)); },
                        [&](const std::monostate &arg) { return unknown(); },
                    }, arg_info.style);
                };

                // TODO(abi): Once `sret` is supported, this will fire.
                VAST_ASSERT(func_info.rets().size() == rets.size());
                zip(func_info.rets(), rets, process);

                return rewriter.create< hl::ReturnOp >( op.getLoc(), out );
            }

            auto emit_all()
            {
                return emit_epilogue( emit_call( emit_prologue() ) );
            }

        };

        struct func_type : OpConversionPattern< mlir::func::FuncOp >
        {
            using Base = OpConversionPattern< mlir::func::FuncOp >;
            using Op = mlir::func::FuncOp;

            TypeConverter &tc;

            func_type(TypeConverter &tc, MContext &mctx)
                : Base(tc, &mctx), tc(tc)
            {}


            // Wrapper has old type and is supposed to be called. Then it converts
            // its arguments and calls the "implementation" which has the correct
            // abi type. Return value of that call is again transformed to the original
            // type.
            abi::WrapFuncOp mk_wrapper(Op op, typename Op::Adaptor ops,
                                       mlir::ConversionPatternRewriter &rewriter) const
            {
                mlir::SmallVector< mlir::DictionaryAttr, 8 > arg_attrs;
                mlir::SmallVector< mlir::NamedAttribute, 8 > other_attrs;

                op.getAllArgAttrs(arg_attrs);
                auto wrapper = rewriter.create< abi::WrapFuncOp >(
                        op.getLoc(),
                        op.getName(),
                        op.getFunctionType(),
                        hl::GlobalLinkageKind::InternalLinkage,
                        other_attrs,
                        arg_attrs
                );

                // Copying visibility from the original function results in error?
                wrapper.setVisibility(mlir::SymbolTable::Visibility::Private);

                auto guard = mlir::OpBuilder::InsertionGuard(rewriter);
                llvm::dbgs() << op.getFunctionType().getInputs().size()
                             << " " << collect_arg_locs(op).size() << "\n";
                auto prologue = rewriter.createBlock(&wrapper.getBody(), {},
                                                     wrapper.getFunctionType().getInputs(),
                                                     collect_arg_locs(op));

                rewriter.setInsertionPointToStart(prologue);

                // This function has old type, we need to emit rules to modify the type now.
                auto converted_type = abi::make_x86_64< Op >(op, tc.dl);
                llvm::dbgs() << converted_type.to_string() << "\n"; llvm::dbgs().flush();

                auto builder = wrapper_builder(op, ops, rewriter,
                                               wrapper, std::move(converted_type));

                builder.emit_all();
                return wrapper;
            }

            mlir::LogicalResult matchAndRewrite(
                    Op op, typename Op::Adaptor ops,
                    mlir::ConversionPatternRewriter &rewriter) const override
            {

                auto wrapper = mk_wrapper(op, ops, rewriter);
                if (!wrapper)
                    return mlir::failure();
                rewriter.eraseOp(op);
                return mlir::success();

            }
        };

    } // namespace pattern


    struct ABIfy : ABIfyBase< ABIfy >
    {
        void runOnOperation() override
        {
            auto &mctx = this->getContext();
            mlir::ModuleOp op = this->getOperation();

            mlir::ConversionTarget target(mctx);
            target.markUnknownOpDynamicallyLegal([](auto) { return true; });
            auto should_transform = [&](mlir::func::FuncOp op)
            {
                // TODO(abi): Due to some issues with location info of arguments
                //            declaration are not yet supported.
                return op.getName() == "main" && !op.isDeclaration();
            };
            target.addDynamicallyLegalOp< mlir::func::FuncOp >(should_transform);

            const auto &dl_analysis = this->getAnalysis< mlir::DataLayoutAnalysis >();

            auto type_converter = TypeConverter(dl_analysis.getAtOrAbove(op), mctx);

            mlir::RewritePatternSet patterns(&mctx);
            patterns.add< pattern::func_type >(type_converter, mctx);
            if (mlir::failed(mlir::applyPartialConversion(op, target, std::move(patterns))))
                return signalPassFailure();

        }
    };

} // namespace vast

std::unique_ptr< mlir::Pass > vast::createABIfyPass()
{
    return std::make_unique< vast::ABIfy >();
}
