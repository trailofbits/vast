// Copyright (c) 2021-present, Trail of Bits, Inc.

#include "vast/Conversion/Passes.hpp"

VAST_RELAX_WARNINGS
#include <mlir/Analysis/DataLayoutAnalysis.h>
#include <mlir/Conversion/LLVMCommon/Pattern.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <llvm/ADT/APFloat.h>
#include <llvm/ADT/SmallVector.h>
VAST_UNRELAX_WARNINGS

#include "../PassesDetails.hpp"

#include "vast/Dialect/HighLevel/HighLevelAttributes.hpp"
#include "vast/Dialect/HighLevel/HighLevelOps.hpp"
#include "vast/Dialect/HighLevel/HighLevelTypes.hpp"

#include "vast/Dialect/Core/CoreAttributes.hpp"
#include "vast/Dialect/Core/CoreOps.hpp"

#include "vast/Dialect/LowLevel/LowLevelOps.hpp"

#include "vast/Dialect/Core/CoreOps.hpp"
#include "vast/Dialect/Core/TypeTraits.hpp"

#include "vast/Util/Common.hpp"
#include "vast/Util/Symbols.hpp"
#include "vast/Util/Terminator.hpp"
#include "vast/Util/TypeList.hpp"

#include "vast/Conversion/Common/Mixins.hpp"
#include "vast/Conversion/TypeConverters/LLVMTypeConverter.hpp"

#include "Common.hpp"
#include "LLCFToLLVM.hpp"

namespace vast::conv::irstollvm
{
    using ignore_patterns = util::type_list<
        ignore_pattern< hl::PredefinedExpr >,
        ignore_pattern< hl::AddressOf >,
        ignore_pattern< hl::NullStmt >,
        erase_pattern< hl::StructDeclOp >,
        erase_pattern< hl::UnionDeclOp >,
        erase_pattern< hl::TypeDeclOp >
    >;

    // `InitListExpr` will construct more complicated types from its elements and if
    // there is some nesting (nested structures for example) there will be a chain of
    // them. This helper recursively traverses such chains and emits appropriate
    // `mlir::LLVM::InsertValueOp` to construct the final value.
    template< typename self_t >
    struct value_builder
    {
      private:
        const self_t &self() const { return static_cast< const self_t & >(*this); }

      protected:

        mlir_value get_only_result(operation op) const {
            VAST_ASSERT(op);
            VAST_CHECK(op->getResults().size() == 1, "Unexpected number of results: {0}", *op);
            return op->getResults()[0];
        }

        mlir_value construct_value(auto &rewriter, mlir_value val) const {
            if (auto bb_arg = mlir::dyn_cast< mlir::BlockArgument >(val))
                return bb_arg;

            auto op = val.getDefiningOp();
            auto trg_type = self().convert(get_only_result(op).getType());
            return construct_value(rewriter, op, trg_type);
        }

        mlir_value construct_value(auto &rewriter, operation op, mlir_type type) const {
            auto init_list = mlir::dyn_cast< hl::InitListExpr >(op);
            if (!init_list) {
                return get_only_result(op);
            }
            return construct_value(rewriter, init_list.getLoc(), init_list.getElements(), type);
        }

        mlir_value construct_value(auto &rewriter, auto loc,
                                   const auto &operands, mlir_type type) const
        {
            auto trg_type = self().convert(type);
            if (mlir::isa< mlir::LLVM::LLVMArrayType, mlir::LLVM::LLVMStructType >(trg_type))
                return construct_aggregate_value(rewriter, loc, operands, trg_type);
            VAST_FATAL("Not implemented yet.");
        }

        mlir_value construct_aggregate_value(
            auto &rewriter, auto loc, const auto &operands,
            mlir_type aggregate_type
        ) const {
            // Currently we are only supporting this type of initialization so
            // better be defensive about it.
            mlir_value init = self().undef(rewriter, loc, aggregate_type);
            for (auto [idx, element] : llvm::enumerate(operands))
            {
                auto elem = construct_value(rewriter, element);
                init = rewriter.template create< mlir::LLVM::InsertValueOp >(
                    loc, init, elem, idx);
            }
            return init;
        }

        mlir_value construct_aggregate_value(
            auto &rewriter, hl::InitListExpr init_list,
            mlir_type aggregate_type
        ) const {
            return construct_aggregate_value(
                rewriter, init_list.getLoc(),
                init_list.getElements(), aggregate_type);
        }
    };


    struct ll_struct_gep : base_pattern< ll::StructGEPOp >
    {
        using base = base_pattern< ll::StructGEPOp >;
        using base::base;

        using op_t = ll::StructGEPOp;

        logical_result matchAndRewrite(
            op_t op, typename op_t::Adaptor ops, conversion_rewriter &rewriter
        ) const override {
            std::vector< mlir::LLVM::GEPArg > indices{ 0ul, ops.getIdx() };

            auto ptr = mlir::dyn_cast< hl::PointerType >(op.getRecord().getType());
            VAST_CHECK(ptr, "{0} is not a pointer to record!", op.getRecord().getType());

            auto gep = rewriter.create< mlir::LLVM::GEPOp >(
                op.getLoc(), convert(op.getType()),
                convert(ptr.getElementType()), ops.getRecord(), indices
            );

            rewriter.replaceOp(op, gep);
            return mlir::success();
        }
    };

    struct ll_extract : base_pattern< ll::Extract >
    {
        using base = base_pattern< ll::Extract >;
        using base::base;

        using op_t = ll::Extract;

        std::size_t to_number(mlir::TypedAttr attr) const {
            auto int_attr = mlir::dyn_cast< mlir::IntegerAttr >(attr);
            VAST_CHECK(int_attr, "Cannot convert {0} to `mlir::IntegerAttr`.", attr);

            return int_attr.getUInt();
        }

        bool is_consistent(op_t op) const {
            auto size      = to_number(op.getTo()) - to_number(op.getFrom()) + 1;
            const auto &dl = this->type_converter().getDataLayoutAnalysis()->getAtOrAbove(op);
            auto target_bw = dl.getTypeSizeInBits(convert(op.getType()));

            return target_bw != size;
        }

        logical_result matchAndRewrite(
            op_t op, typename op_t::Adaptor ops, conversion_rewriter &rewriter
        ) const override {
            auto loc = op.getLoc();

            auto value = ops.getArg();
            auto from = op.from();

            auto shift = rewriter.create< mlir::LLVM::LShrOp >(
                loc, value, iN(rewriter, loc, value.getType(), from)
            );
            auto trg_type = convert(op.getType());
            auto trunc = rewriter.create< mlir::LLVM::TruncOp >(loc, trg_type, shift);
            rewriter.replaceOp(op, trunc);
            return mlir::success();
        }
    };

    struct ll_concat : base_pattern< ll::Concat >
    {
        using base = base_pattern< ll::Concat >;
        using base::base;

        using op_t = ll::Concat;

        std::size_t bw(operation op) const {
            VAST_ASSERT(op->getNumResults() == 1);
            const auto &dl = this->type_converter().getDataLayoutAnalysis()->getAtOrAbove(op);
            return dl.getTypeSizeInBits(convert(op->getResult(0).getType()));
        }

        logical_result matchAndRewrite(
            op_t op, typename op_t::Adaptor ops, conversion_rewriter &rewriter
        ) const override {
            auto loc = op.getLoc();

            auto resize = [&](auto w) -> mlir::Value {
                auto trg_type = convert(op.getType());
                if (w.getType() == trg_type) {
                    return w;
                }
                return rewriter.create< mlir::LLVM::ZExtOp >(loc, trg_type, w);
            };
            mlir::Value head = resize(ops.getOperands()[0]);

            std::size_t start = bw(ops.getOperands()[0].getDefiningOp());
            for (std::size_t i = 1; i < ops.getOperands().size(); ++i) {
                auto full    = resize(ops.getOperands()[i]);
                auto shifted = rewriter.create< mlir::LLVM::ShlOp >(
                    loc, full, mk_index(loc, start, rewriter)
                );
                head = rewriter.create< mlir::LLVM::OrOp >(loc, head, shifted->getResult(0));

                start += bw(ops.getOperands()[i].getDefiningOp());
            }

            rewriter.replaceOp(op, head);
            return mlir::success();
        }
    };

    using ll_generic_patterns = util::type_list<
        ll_struct_gep,
        ll_extract,
        ll_concat
    >;

    template< typename Op >
    struct inline_region_from_op : base_pattern< Op >
    {
        using base = base_pattern< Op >;
        using base::base;

        logical_result matchAndRewrite(
            Op unit_op, typename Op::Adaptor ops, conversion_rewriter &rewriter
        ) const override {
            auto parent = unit_op.getBody().getParentRegion();
            rewriter.inlineRegionBefore(unit_op.getBody(), *parent, parent->end());

            // splice newly created translation unit block in the module
            auto &unit_block = parent->back();
            rewriter.mergeBlocks(&parent->front(), &unit_block, unit_block.getArguments());

            rewriter.eraseOp(unit_op);
            return logical_result::success();
        }
    };

    template< typename op_t >
    struct hl_scopelike : ll_cf::scope_like< op_t >
    {
        using base = ll_cf::scope_like< op_t >;
        using base::base;

        using adaptor_t = typename op_t::Adaptor;

        mlir::Block *start_block(op_t op) const override { return &op.getBody().front(); }

        auto matchAndRewrite(op_t op, adaptor_t ops, conversion_rewriter &rewriter) const
            -> logical_result override
        {
            if (op.getBody().empty()) {
                rewriter.eraseOp(op);
                return logical_result::success();
            }

            // If we do not have any branching inside, we can just "inline" the op.
            if (op.getBody().hasOneBlock()) {
                return base::handle_singleblock(op, ops, rewriter);
            }

            return base::handle_multiblock(op, ops, rewriter);
        }

        static void legalize(conversion_target &trg)
        {
            trg.addIllegalOp< op_t >();
        }
    };

    using label_stmt = hl_scopelike< hl::LabelStmt >;
    using scope_op   = hl_scopelike< core::ScopeOp >;

    using label_patterns = util::type_list< erase_pattern< hl::LabelDeclOp >, label_stmt >;

    // TODO(conv): Figure out if these can be somehow unified.
    using inline_region_from_op_conversions =
        util::type_list< inline_region_from_op< hl::TranslationUnitOp >, scope_op >;

    template< typename op_t >
    struct subscript_like : base_pattern< op_t >
    {
        using Op = op_t;
        using base = base_pattern< Op >;
        using base::base;

        logical_result matchAndRewrite(
                Op op, typename Op::Adaptor ops,
                conversion_rewriter &rewriter) const override
        {
            auto trg_type = this->convert(op.getType());
            auto element_type = this->converted_element_type(op.getType());
            VAST_PATTERN_CHECK(trg_type, "Could not convert vardecl type");

            auto gep = rewriter.create< mlir::LLVM::GEPOp >(
                    op.getLoc(),
                    trg_type, element_type,
                    ops.getArray(), ops.getIndex() );

            rewriter.replaceOp(op, gep);
            return logical_result::success();
        }
    };

    struct init_list_expr : base_pattern< hl::InitListExpr >,
                            value_builder< init_list_expr >
    {
        using op_t = hl::InitListExpr;
        using base = base_pattern< op_t >;
        using base::base;

        logical_result matchAndRewrite(
                op_t op, typename op_t::Adaptor ops,
                conversion_rewriter &rewriter) const override
        {
            VAST_PATTERN_CHECK(op.getNumResults() == 1, "Unexpected number of results");
            auto value = construct_value(
                rewriter, op.getLoc(), ops.getOperands(), op.getType(0));
            rewriter.replaceOp(op, value);
            return mlir::success();
        }
    };

    struct vardecl : base_pattern< hl::VarDeclOp >
    {
        using op_t = hl::VarDeclOp;
        using base = base_pattern< op_t >;
        using base::base;

        logical_result matchAndRewrite(
                op_t op, typename op_t::Adaptor ops,
                conversion_rewriter &rewriter) const override
        {
            auto t = mlir::dyn_cast< hl::PointerType >(op.getType());
            auto target_type = this->convert(t.getElementType());

            // Sadly, we cannot build `mlir::LLVM::GlobalOp` without
            // providing a value attribute.
            auto create_dummy_value = [&] () -> mlir::Attribute {
                if (auto trg_arr = mlir::dyn_cast< mlir::LLVM::LLVMArrayType >(target_type)) {
                    attrs_t arr(trg_arr.getNumElements(),
                                rewriter.getIntegerAttr(rewriter.getIndexType(), 0));
                    return rewriter.getArrayAttr(arr);
                }
                return rewriter.getIntegerAttr(rewriter.getIndexType(), 0);
            };

            // So we know this is a global, otherwise it would be in `ll:`.
            auto gop = rewriter.create< mlir::LLVM::GlobalOp >(
                    op.getLoc(),
                    target_type,
                    // TODO(conv:irstollvm): Constant.
                    true,
                    LLVM::Linkage::Internal,
                    op.getSymbolName(), create_dummy_value());

            // If we want the global to have a body it cannot have value attribute.
            gop.removeValueAttr();

            // We could probably try to analyze the region to see if it isn't
            // a case where we can just do an attribute, but for now let's
            // just use the initializer.
            auto &region = gop.getInitializerRegion();
            rewriter.inlineRegionBefore(op.getInitializer(),
                                        region, region.begin());
            rewriter.eraseOp(op);
            return logical_result::success();
        }

    };

    struct global_ref : base_pattern< hl::DeclRefOp >
    {
        using op_t = hl::DeclRefOp;
        using base = base_pattern< op_t >;
        using base::base;

        logical_result matchAndRewrite(
                op_t op, typename op_t::Adaptor ops,
                conversion_rewriter &rewriter) const override
        {
            auto target_type = this->convert(op.getType());

            auto addr_of = rewriter.template create< mlir::LLVM::AddressOfOp >(
                    op.getLoc(),
                    target_type,
                    op.getName());
            rewriter.replaceOp(op, addr_of);
            return logical_result::success();
        }

    };

    using init_conversions = util::type_list<
        init_list_expr,
        vardecl,
        global_ref
    >;

    template< typename op_t >
    struct func_op : base_pattern< op_t >
    {
        using base = base_pattern< op_t >;
        using base::base;

        using adaptor_t = typename op_t::Adaptor;

        using llvm_func_op = mlir::LLVM::LLVMFuncOp;

        logical_result matchAndRewrite(
            op_t func_op, adaptor_t ops, conversion_rewriter &rewriter
        ) const override {
            auto &tc = this->type_converter();
            auto maybe_target_type = tc.convert_fn_t(func_op.getFunctionType());
            // TODO(irs-to-llvm): Handle varargs.
            auto maybe_signature = tc.get_conversion_signature(func_op, /* variadic */ true);

            // Type converter failed.
            if (!maybe_target_type || !*maybe_target_type || !maybe_signature) {
                VAST_PATTERN_FAIL(
                    "Failed to convert function type: {0}", func_op.getFunctionType()
                );
            }

            auto target_type = *maybe_target_type;
            auto signature = *maybe_signature;

            // TODO(irs-to-llvm): Currently it is unclear what to do with the
            //                    arg/res attributes as it looks like we may not
            //                    want to lower them all.


            // TODO(lukas): Linkage?
            auto linkage = LLVM::Linkage::External;
            auto new_func = rewriter.create< llvm_func_op >(
                func_op.getLoc(), func_op.getName(),
                target_type, linkage,
                func_op.isVarArg(), LLVM::CConv::C
            );

            rewriter.inlineRegionBefore(func_op.getBody(), new_func.getBody(), new_func.end());
            tc::convert_region_types(func_op, new_func, signature);

            if (mlir::failed(args_to_allocas(new_func, rewriter))) {
                VAST_PATTERN_FAIL("Failed to convert func arguments");
            }

            rewriter.eraseOp(func_op);
            return logical_result::success();
        }

        logical_result args_to_allocas(
            llvm_func_op fn, conversion_rewriter &rewriter
        ) const {
            if (fn.empty())
                return logical_result::success();

            auto &block = fn.front();
            if (!block.isEntryBlock())
                return logical_result::failure();

            return logical_result::success();
        }

        static void legalize(conversion_target &target) { target.addIllegalOp< op_t >(); }
    };

    struct constant_int : base_pattern< hl::ConstantOp >
    {
        using base = base_pattern< hl::ConstantOp >;
        using base::base;

        static inline constexpr const char *strlit_global_var_prefix = "vast.strlit.constant_";

        std::string next_strlit_name(core::module mod) const
        {
            std::size_t current = 0;
            for (auto &op : mod.getOps())
            {
                auto global = mlir::dyn_cast< mlir::LLVM::GlobalOp >(op);
                if (!global)
                    continue;

                auto name = global.getName();
                if (!name.consume_front(strlit_global_var_prefix))
                    continue;

                std::size_t idx = 0;
                name.getAsInteger(10, idx);
                current = std::max< std::size_t >(idx, current);
            }
            return strlit_global_var_prefix + std::to_string(current + 1);

        }

        logical_result handle_void_const(
                hl::ConstantOp op, conversion_rewriter &rewriter) const
        {
            auto result = op.getResult();
            if (result.hasOneUse()) {
                auto user = result.getUses().begin()->getOwner();
                if (core::is_return_like(user)) {
                    auto guard = insertion_guard(rewriter);
                    rewriter.setInsertionPoint(user);

                    rewriter.create< LLVM::ReturnOp >(user->getLoc(), mlir::ValueRange());
                    rewriter.eraseOp(user);
                    rewriter.eraseOp(op);
                    return logical_result::success();
                }
            }
            return logical_result::failure();

        }

        logical_result matchAndRewrite(
                hl::ConstantOp op, hl::ConstantOp::Adaptor ops,
                conversion_rewriter &rewriter) const override
        {
            if (mlir::isa< mlir::NoneType >(op.getResult().getType())) {
                return handle_void_const(op, rewriter);
            }

            auto val = make_from(op, rewriter, this->type_converter());
            if (!val)
                return mlir::failure();

            rewriter.replaceOp(op, val);
            return logical_result::success();
        }

        mlir::Attribute convert_attr(auto attr, auto op,
                                     conversion_rewriter &rewriter) const
        {
            auto target_type = this->convert(attr.getType());
            const auto &dl = this->type_converter().getDataLayoutAnalysis()
                                                   ->getAtOrAbove(op);
            if (!target_type)
                return {};

            if (auto float_attr = attr.template dyn_cast< core::FloatAttr >())
            {
                // NOTE(lukas): We cannot simply forward the return value of `getValue()`
                //              because it can have different semantics than one expected
                //              by `FloatAttr`.
                // TODO(lukas): Is there a better way to convert this?
                //              Ideally `APFloat -> APFloat`.
                double raw_value = float_attr.getValue().convertToDouble();
                return rewriter.getFloatAttr(target_type, raw_value);
            }

            if (auto int_attr = attr.template dyn_cast< core::IntegerAttr >())
            {
                auto size = dl.getTypeSizeInBits(target_type);
                auto coerced = int_attr.getValue().sextOrTrunc(size);
                return rewriter.getIntegerAttr(target_type, coerced);
            }

            VAST_UNREACHABLE("Trying to convert attr that is not supported, {0} in op {1}",
                             attr, op);
            return {};
        }


        mlir::Value convert_strlit(hl::ConstantOp op, auto rewriter, auto &tc,
                                   mlir_type original_type, mlir::StringAttr str_lit) const
        {
            // We need to include the terminating `0` which will not happen
            // if we "just" pass the value in.
            auto twine = llvm::Twine(std::string_view(str_lit.getValue().data(),
                                                      str_lit.getValue().size() + 1));
            auto converted_attr = mlir::StringAttr::get(op->getContext(), twine);

            auto mod = op->getParentOfType< core::module >();
            auto name = next_strlit_name(mod);

            auto et = converted_element_type(original_type);
            if (!et)
                return {};

            rewriter.guarded([&]()
            {
                rewriter->setInsertionPoint(&*mod.begin());
                rewriter->template create< mlir::LLVM::GlobalOp >(
                    op.getLoc(),
                    et,
                    true, /* is constant */
                    LLVM::Linkage::Internal,
                    name,
                    converted_attr);
            });

            return rewriter->template create< mlir::LLVM::AddressOfOp >(
                    op.getLoc(), convert(original_type), name);
        }

        mlir::Value make_from(
                hl::ConstantOp op,
                conversion_rewriter &rewriter,
                auto &&tc) const
        {
            auto target_type = this->convert(op.getType());

            if (auto str_lit = mlir::dyn_cast< mlir::StringAttr >(op.getValue()))
                return convert_strlit(op, rewriter_wrapper_t(rewriter), tc,
                                      op.getType(), str_lit);

            auto attr = convert_attr(op.getValue(), op, rewriter);
            if (!attr)
                return {};
            return rewriter.create< LLVM::ConstantOp >(op.getLoc(), target_type, attr);
        }
    };

    template< typename return_like_op >
    struct ret : base_pattern< return_like_op >
    {
        using op_t = return_like_op;
        using base = base_pattern< op_t >;
        using base::base;

        logical_result matchAndRewrite(
                op_t ret_op, typename op_t::Adaptor ops,
                conversion_rewriter &rewriter) const override
        {
            rewriter.create< LLVM::ReturnOp >(
                ret_op.getLoc(), this->filter_out_void(ops.getOperands()));
            rewriter.eraseOp(ret_op);
            return logical_result::success();
        }

        static void legalize(conversion_target &target)
        {
            target.addIllegalOp< return_like_op >();
        }
    };

    using return_conversions = util::type_list<
          ret< hl::ReturnOp >
        , ret< ll::ReturnOp >
        , ret< mlir::func::ReturnOp >
        // Special case is if void is returned. Since `core.void` is actually a value
        // we need to handle it when lowering constants. For now we also handle return there
        // but we should consider whether instead not ignore the constant if it is void and
        // let the users handle that special case.
        , ret< core::ImplicitReturnOp >
    >;

    logical_result match_and_rewrite_cast(auto &pattern, auto op, auto adaptor, auto &rewriter) {
        auto dst_type = pattern.convert(op.getType());
        auto element_type = pattern.converted_element_type(op.getType());

        auto src = adaptor.getValue();
        auto src_type = src.getType();

        auto orig_dst_type = op.getType();
        auto orig_src_type = op.getValue().getType();

        auto bitcast = [&] {
            rewriter.template replaceOpWithNewOp< LLVM::BitcastOp >(op, dst_type, src);
            return mlir::success();
        };

        auto lvalue_to_rvalue = [&] {
            rewriter.template replaceOpWithNewOp< LLVM::LoadOp >(op, dst_type, src);
            return mlir::success();
        };

        auto noop = [&] {
            rewriter.replaceOp(op, { src });
            return mlir::success();
        };

        auto arr_to_ptr_decay = [&] {
            rewriter.template replaceOpWithNewOp< LLVM::GEPOp >(
                op, dst_type, element_type, src, LLVM::GEPArg{ 0 }
            );
            return mlir::success();
        };

        auto null_to_ptr = [&] {
            // TODO: deal with nullptr_t conversion
            auto null = pattern.null_ptr(rewriter, op.getLoc(), dst_type);
            rewriter.replaceOp(op, { null });
            return mlir::success();
        };

        auto int_to_ptr = [&] {
            rewriter.template replaceOpWithNewOp< LLVM::IntToPtrOp >(op, dst_type, src);
            return mlir::success();
        };

        auto ptr_to_int = [&] {
            rewriter.template replaceOpWithNewOp< LLVM::PtrToIntOp >(op, dst_type, src);
            return mlir::success();
        };

        auto ptr_to_bool = [&] {
            auto null = pattern.constant(rewriter, op.getLoc(), src_type, 0);
            rewriter.template replaceOpWithNewOp< hl::CmpOp >(
                op, dst_type, hl::Predicate::ne, src, null
            );
            return mlir::success();
        };

        auto to_void = [&] {
            rewriter.replaceOp(op, { src });
            return mlir::success();
        };

        auto integral_cast = [&] {
            // TODO: consult with clang, we are mistreating bool -> int conversion
            pattern.replace_with_trunc_or_ext(op, src, orig_src_type, dst_type, rewriter);
            return mlir::success();
        };

        auto int_to_bool = [&] {
            auto zero = pattern.constant(rewriter, op.getLoc(), src_type, 0);
            rewriter.template replaceOpWithNewOp< hl::CmpOp >(
                op, dst_type, hl::Predicate::ne, src, zero
            );
            return mlir::success();
        };

        auto int_to_float = [&] {
            if (orig_src_type.isSignedInteger()) {
                rewriter.template replaceOpWithNewOp< LLVM::SIToFPOp >(op, dst_type, src);
            } else {
                rewriter.template replaceOpWithNewOp< LLVM::UIToFPOp >(op, dst_type, src);
            }
            return mlir::success();
        };

        auto float_to_int = [&] {
            if (orig_dst_type.isSignedInteger()) {
                rewriter.template replaceOpWithNewOp< LLVM::FPToSIOp >(op, dst_type, src);
            } else {
                rewriter.template replaceOpWithNewOp< LLVM::FPToUIOp >(op, dst_type, src);
            }
            return mlir::success();
        };

        auto float_to_bool = [&] {
            // Check if float is not equal to zero.
            auto zero = pattern.constant(rewriter, op.getLoc(), src_type, 0.0);
            auto cmp = rewriter.template create< LLVM::FCmpOp >(
                op.getLoc(), LLVM::FCmpPredicate::une, src, zero
            );

            // Extend comparison result to either bool (C++) or int (C).
            rewriter.template replaceOpWithNewOp< LLVM::ZExtOp >(op, dst_type, cmp);
            return mlir::success();
        };

        auto bool_to_int = [&] {
            auto trunc = rewriter.template create< LLVM::TruncOp >(
                op.getLoc(), mlir::IntegerType::get(op.getContext(), 1), src
            );
            rewriter.template replaceOpWithNewOp< LLVM::ZExtOp >(op, dst_type, trunc);
            return mlir::success();
        };

        auto floating_cast = [&] {
            auto src_bw = src_type.getIntOrFloatBitWidth();
            auto dst_bw = dst_type.getIntOrFloatBitWidth();

            if (src_bw > dst_bw) {
                rewriter.template replaceOpWithNewOp< LLVM::FPTruncOp >(op , dst_type, src);
            } else {
                rewriter.template replaceOpWithNewOp< LLVM::FPExtOp >(op, dst_type, src);
            }
            return mlir::success();
        };

        switch (op.getKind()) {
            case hl::CastKind::BitCast:
                return bitcast();
            // case hl::CastKind::LValueBitCast:
            // case hl::CastKind::LValueToRValueBitCast:
            case hl::CastKind::LValueToRValue:
                return lvalue_to_rvalue();

            case hl::CastKind::NoOp:
                return noop();

            // case hl::CastKind::BaseToDerived:
            // case hl::CastKind::DerivedToBase:
            // case hl::CastKind::UncheckedDerivedToBase:
            // case hl::CastKind::Dynamic:
            // case hl::CastKind::ToUnion:

            case hl::CastKind::ArrayToPointerDecay:
                return arr_to_ptr_decay();
            // case hl::CastKind::FunctionToPointerDecay:
            case hl::CastKind::NullToPointer:
                return null_to_ptr();
            // case hl::CastKind::NullToMemberPointer:
            // case hl::CastKind::BaseToDerivedMemberPointer:
            // case hl::CastKind::DerivedToBaseMemberPointer:
            // case hl::CastKind::MemberPointerToBoolean:
            // case hl::CastKind::UserDefinedConversion:
            // case hl::CastKind::ConstructorConversion:

            case hl::CastKind::IntegralToPointer:
                return int_to_ptr();
            case hl::CastKind::PointerToIntegral:
                return ptr_to_int();
            case hl::CastKind::PointerToBoolean:
                return ptr_to_bool();

            case hl::CastKind::ToVoid:
                return to_void();

            // case hl::CastKind::VectorSplat:
            case hl::CastKind::IntegralCast:
                return integral_cast();
            case hl::CastKind::IntegralToBoolean:
                return int_to_bool();
            case hl::CastKind::IntegralToFloating:
                return int_to_float();
            // case hl::CastKind::FloatingToFixedPoint:
            // case hl::CastKind::FixedPointToFloating:
            // case hl::CastKind::FixedPointCast:
            // case hl::CastKind::FixedPointToIntegral:
            // case hl::CastKind::IntegralToFixedPoint:
            // case hl::CastKind::FixedPointToBoolean:
            case hl::CastKind::FloatingToIntegral:
                return float_to_int();
            case hl::CastKind::FloatingToBoolean:
                return float_to_bool();
            case hl::CastKind::BooleanToSignedIntegral:
                return bool_to_int();
            case hl::CastKind::FloatingCast:
                return floating_cast();

            // case hl::CastKind::CPointerToObjCPointerCast:
            // case hl::CastKind::BlockPointerToObjCPointerCast:
            // case hl::CastKind::AnyPointerToBlockPointerCast:
            // case hl::CastKind::ObjCObjectLValueCast:

            // case hl::CastKind::FloatingRealToComplex:
            // case hl::CastKind::FloatingComplexToReal:
            // case hl::CastKind::FloatingComplexToBoolean:
            // case hl::CastKind::FloatingComplexCast:
            // case hl::CastKind::FloatingComplexToIntegralComplex:

            // case hl::CastKind::IntegralRealToComplex:
            // case hl::CastKind::IntegralComplexToReal:
            // case hl::CastKind::IntegralComplexToBoolean:
            // case hl::CastKind::IntegralComplexCast:
            // case hl::CastKind::IntegralComplexToFloatingComplex:

            // case hl::CastKind::ARCProduceObject:
            // case hl::CastKind::ARCConsumeObject:
            // case hl::CastKind::ARCReclaimReturnedObject:
            // case hl::CastKind::ARCExtendBlockObject:

            // case hl::CastKind::AtomicToNonAtomic:
            // case hl::CastKind::NonAtomicToAtomic:

            // case hl::CastKind::CopyAndAutoreleaseBlockObject:
            // case hl::CastKind::BuiltinFnToFnPtr:
            // case hl::CastKind::ZeroToOCLOpaqueType:
            // case hl::CastKind::AddressSpaceConversion:
            // case hl::CastKind::IntToOCLSampler:
            // case hl::CastKind::MatrixCast:
            default:
                return logical_result::failure();
        }
        return logical_result::success();
    }


    struct implicit_cast : base_pattern< hl::ImplicitCastOp >
    {
        using op_t = hl::ImplicitCastOp;
        using base = base_pattern< hl::ImplicitCastOp >;
        using base::base;

        using adaptor_t = typename op_t::Adaptor;

        logical_result matchAndRewrite(
            op_t op, adaptor_t adaptor,
            conversion_rewriter &rewriter
        ) const override {
            return match_and_rewrite_cast(*this, op, adaptor, rewriter);
        }
    };

    struct cstyle_cast : base_pattern< hl::CStyleCastOp >
    {
        using op_t = hl::CStyleCastOp;
        using base = base_pattern< op_t >;
        using base::base;

        using adaptor_t = typename op_t::Adaptor;

        logical_result matchAndRewrite(
            op_t op, adaptor_t adaptor,
            conversion_rewriter &rewriter
        ) const override {
            return match_and_rewrite_cast(*this, op, adaptor, rewriter);
        }
    };

    template< typename src_t, typename trg_t >
    struct shift_op : base_pattern< src_t >
    {
        using base = base_pattern< src_t >;
        using base::base;

        using adaptor_t = typename src_t::Adaptor;

        logical_result
        matchAndRewrite(src_t op, adaptor_t ops, conversion_rewriter &rewriter) const override {
            auto target_ty = this->type_converter().convert_type_to_type(op.getType());
            auto rhs       = base::insert_trunc_or_ext(ops.getRhs(), *target_ty, rewriter);
            auto new_op = rewriter.create< trg_t >(op.getLoc(), *target_ty, ops.getLhs(), rhs);
            rewriter.replaceOp(op, new_op);
            return mlir::success();
        }
    };

    using shift_conversions = util::type_list<
        shift_op< hl::BinShlOp, LLVM::ShlOp >,

        shift_op< hl::BinLShrOp, LLVM::LShrOp >,
        shift_op< hl::BinAShrOp, LLVM::AShrOp >
    >;

    using one_to_one_conversions = util::type_list<
        one_to_one< hl::AddIOp, LLVM::AddOp >,
        one_to_one< hl::SubIOp, LLVM::SubOp >,
        one_to_one< hl::MulIOp, LLVM::MulOp >,

        one_to_one< hl::AddFOp, LLVM::FAddOp >,
        one_to_one< hl::SubFOp, LLVM::FSubOp >,
        one_to_one< hl::MulFOp, LLVM::FMulOp >,

        one_to_one< hl::DivSOp, LLVM::SDivOp >,
        one_to_one< hl::DivUOp, LLVM::UDivOp >,
        one_to_one< hl::DivFOp, LLVM::FDivOp >,

        one_to_one< hl::RemSOp, LLVM::SRemOp >,
        one_to_one< hl::RemUOp, LLVM::URemOp >,
        one_to_one< hl::RemFOp, LLVM::FRemOp >,

        one_to_one< hl::BinOrOp, LLVM::OrOp >,
        one_to_one< hl::BinAndOp, LLVM::AndOp >,
        one_to_one< hl::BinXorOp, LLVM::XOrOp >
    >;

    struct call : base_pattern< hl::CallOp >
    {
        using op_t = hl::CallOp;
        using base = base_pattern< op_t >;
        using base::base;

        using adaptor_t = typename op_t::Adaptor;

        logical_result matchAndRewrite(
            op_t op, adaptor_t ops, conversion_rewriter &rewriter
        ) const override {
            auto caller = mlir::dyn_cast< VastCallOpInterface >(op.getOperation());
            if (!caller) {
                return logical_result::failure();
            }

            auto callee = caller.resolveCallable();
            if (!callee && !mlir::isa< mlir::FunctionOpInterface >(callee)) {
                return logical_result::failure();
            }

            auto fn = mlir::cast< mlir::FunctionOpInterface >(callee);
            auto rtys = type_converter().convert_types_to_types(fn.getResultTypes());

            if (!rtys) {
                return logical_result::failure();
            }

            auto mk_call = [&](auto... args) {
                return rewriter.create< mlir::LLVM::CallOp >(op.getLoc(), args...);
            };

            if (rtys->empty() || rtys->front().isa< mlir::LLVM::LLVMVoidType >()) {
                // We cannot pass in void type as some internal check inside `mlir::LLVM`
                // dialect will fire - it would create a value of void type, which is not
                // allowed.
                mk_call(types_t{}, op.getCallee(), ops.getOperands());
                rewriter.eraseOp(op);
            } else {
                auto call = mk_call(*rtys, op.getCallee(), ops.getOperands());
                rewriter.replaceOp(op, call.getResults());
            }

            return logical_result::success();
        }
    };

    bool is_lvalue(auto op)
    {
        return op && op.getType().template isa< hl::LValueType >();
    }

    struct logical_not : base_pattern< hl::LNotOp >
    {
        using base = base_pattern< hl::LNotOp >;
        using base::base;
        using adaptor_t = typename hl::LNotOp::Adaptor;
        using integer_t = mlir::IntegerType;

        Operation * create_cmp(conversion_rewriter &rewriter, hl::LNotOp &op, Value lhs, Value rhs) const {
            auto lhs_type = lhs.getType();

            if (llvm::isa< mlir::FloatType >(lhs_type)) {
                auto i1_type = integer_t::get(op.getContext(), 1, integer_t::Signless);

                return rewriter.create< LLVM::FCmpOp >(
                            op.getLoc(), i1_type,
                            LLVM::FCmpPredicate::une,
                            lhs, rhs
                       );
            }

            if (llvm::isa< mlir::IntegerType >(lhs_type)) {
                return rewriter.create< LLVM::ICmpOp >(
                    op.getLoc(),
                    LLVM::ICmpPredicate::ne,
                    lhs, rhs
                );
            }
            VAST_UNREACHABLE("Unknwon cmp type.");
        }

        logical_result matchAndRewrite(hl::LNotOp op, adaptor_t adaptor, conversion_rewriter &rewriter) const override {
            auto zero = this->constant(rewriter, op.getLoc(), adaptor.getArg().getType(), 0);

            auto cmp = create_cmp(rewriter, op, adaptor.getArg(), zero);

            auto true_i1 = this->iN(rewriter, op.getLoc(), cmp->getResult(0).getType(), 1);
            auto xor_val = rewriter.create< LLVM::XOrOp >(op.getLoc(), cmp->getResult(0), true_i1);

            auto res_type = this->convert(op.getResult().getType());
            if (!res_type)
                return logical_result::failure();
            if (res_type != xor_val.getType()) {
                rewriter.replaceOpWithNewOp< LLVM::ZExtOp >(op, res_type, xor_val);
            } else {
                rewriter.replaceOp(op, xor_val);
            }
            return logical_result::success();
        }

    };

    struct bin_not : base_pattern< hl::NotOp >
    {
        using base = base_pattern< hl::NotOp >;
        using base::base;
        using adaptor_t = typename hl::NotOp::Adaptor;

        logical_result matchAndRewrite(
                hl::NotOp op, adaptor_t adaptor,
                conversion_rewriter &rewriter) const override
        {
            auto helper = this->constant(rewriter, op.getLoc(), adaptor.getArg().getType(), -1);
            rewriter.replaceOpWithNewOp< LLVM::XOrOp >(op, adaptor.getArg(), helper);
            return logical_result::success();
        }
    };

    using unary_in_place_conversions = util::type_list<
        logical_not,
        bin_not
    >;

    struct minus : base_pattern< hl::MinusOp >
    {
        using base = base_pattern< hl::MinusOp >;
        using base::base;

        logical_result matchAndRewrite(
                    hl::MinusOp op, hl::MinusOp::Adaptor adaptor,
                    conversion_rewriter &rewriter) const override
        {
            auto arg = adaptor.getArg();
            if (is_lvalue(arg))
                return logical_result::failure();
            auto arg_type = convert(arg.getType());

            auto zero = this->constant(rewriter, op.getLoc(), arg_type, 0);

            if (llvm::isa< mlir::FloatType >(arg_type))
                rewriter.replaceOpWithNewOp< LLVM::FSubOp >(op, zero, arg);
            else
                rewriter.replaceOpWithNewOp< LLVM::SubOp >(op, zero, arg);

            return logical_result::success();
        }
    };

    using sign_conversions = util::type_list<
        minus,
        ignore_pattern< hl::PlusOp >
    >;

    template<
        typename src_op_t, typename trg_op_t, typename src_predicate_t,
        typename trg_predicate_t >
    struct cmp_base : base_pattern< src_op_t >
    {
        using op_t = src_op_t;
        using base = base_pattern< op_t >;
        using base::base;

        using adaptor_t = typename op_t::Adaptor;

        virtual trg_predicate_t convert_predicate(src_predicate_t predicate) const = 0;

        logical_result matchAndRewrite(
            op_t op, adaptor_t adaptor, conversion_rewriter &rewriter
        ) const override {
            auto pred = convert_predicate(op.getPredicate());

            auto new_cmp = rewriter.create< trg_op_t >(
                op.getLoc(), pred, adaptor.getLhs(), adaptor.getRhs()
            );

            auto dst_type = this->convert(op.getType());
            this->replace_with_trunc_or_ext(op, new_cmp, new_cmp.getType(), dst_type, rewriter);

            return mlir::success();
        }
    };

    using icmp_base = cmp_base< hl::CmpOp, LLVM::ICmpOp, hl::Predicate, LLVM::ICmpPredicate >;
    struct icmp : icmp_base
    {
        using base = icmp_base;
        using base::base;

        mlir::LLVM::ICmpPredicate convert_predicate(hl::Predicate predicate) const override {
            switch (predicate) {
                case hl::Predicate::eq:
                    return LLVM::ICmpPredicate::eq;
                case hl::Predicate::ne:
                    return LLVM::ICmpPredicate::ne;
                case hl::Predicate::slt:
                    return LLVM::ICmpPredicate::slt;
                case hl::Predicate::sle:
                    return LLVM::ICmpPredicate::sle;
                case hl::Predicate::sgt:
                    return LLVM::ICmpPredicate::sgt;
                case hl::Predicate::sge:
                    return LLVM::ICmpPredicate::sge;
                case hl::Predicate::ult:
                    return LLVM::ICmpPredicate::ult;
                case hl::Predicate::ule:
                    return LLVM::ICmpPredicate::ule;
                case hl::Predicate::ugt:
                    return LLVM::ICmpPredicate::ugt;
                case hl::Predicate::uge:
                    return LLVM::ICmpPredicate::uge;
            }
        }
    };

    using fcmp_base = cmp_base< hl::FCmpOp, LLVM::FCmpOp, hl::FPredicate, LLVM::FCmpPredicate >;
    struct fcmp : fcmp_base
    {
        using base = fcmp_base;
        using base::base;

        mlir::LLVM::FCmpPredicate convert_predicate(hl::FPredicate predicate) const override {
            switch (predicate) {
                case hl::FPredicate::ffalse:
                    return LLVM::FCmpPredicate::_false;
                case hl::FPredicate::oeq:
                    return LLVM::FCmpPredicate::oeq;
                case hl::FPredicate::ogt:
                    return LLVM::FCmpPredicate::ogt;
                case hl::FPredicate::oge:
                    return LLVM::FCmpPredicate::oge;
                case hl::FPredicate::olt:
                    return LLVM::FCmpPredicate::olt;
                case hl::FPredicate::ole:
                    return LLVM::FCmpPredicate::ole;
                case hl::FPredicate::one:
                    return LLVM::FCmpPredicate::one;
                case hl::FPredicate::ord:
                    return LLVM::FCmpPredicate::ord;
                case hl::FPredicate::uno:
                    return LLVM::FCmpPredicate::uno;
                case hl::FPredicate::ueq:
                    return LLVM::FCmpPredicate::ueq;
                case hl::FPredicate::ugt:
                    return LLVM::FCmpPredicate::ugt;
                case hl::FPredicate::uge:
                    return LLVM::FCmpPredicate::uge;
                case hl::FPredicate::ult:
                    return LLVM::FCmpPredicate::ult;
                case hl::FPredicate::ule:
                    return LLVM::FCmpPredicate::ule;
                case hl::FPredicate::une:
                    return LLVM::FCmpPredicate::une;
                case hl::FPredicate::ftrue:
                    return LLVM::FCmpPredicate::_true;
            }
        }
    };

    struct deref : base_pattern< hl::Deref >
    {
        using base = base_pattern< hl::Deref >;
        using base::base;

        logical_result matchAndRewrite(
                    hl::Deref op, typename hl::Deref::Adaptor ops,
                    conversion_rewriter &rewriter) const override
        {
            auto trg_type = tc.convert_type_to_type(op.getType());
            if (!trg_type)
                return logical_result::failure();

            auto loaded = rewriter.create< mlir::LLVM::LoadOp >(
                    op.getLoc(), *trg_type, ops.getAddr());
            rewriter.replaceOp(op, loaded);

            return logical_result::success();
        }
    };

    template< typename op_t, typename yield_op_t >
    struct propagate_yield : base_pattern< op_t >
    {
        using base = base_pattern< op_t >;
        using base::base;

        logical_result matchAndRewrite(
                    op_t op, typename op_t::Adaptor ops,
                    conversion_rewriter &rewriter) const override
        {
            auto body = op.getBody();
            if (!body)
                return logical_result::success();

            auto yield = terminator_t< yield_op_t >::get(*body);
            VAST_PATTERN_CHECK(yield, "Expected yield in: {0}", op);

            rewriter.inlineBlockBefore(body, op);
            rewriter.replaceOp(op, yield.op().getResult());
            rewriter.eraseOp(yield.op());
            return logical_result::success();
        }
    };

    struct value_yield_in_global_var : base_pattern< hl::ValueYieldOp >,
                                       value_builder< value_yield_in_global_var >
    {
        using op_t = hl::ValueYieldOp;
        using base = base_pattern< op_t >;
        using base::base;


        logical_result matchAndRewrite(
                    op_t op, typename op_t::Adaptor ops,
                    conversion_rewriter &rewriter) const override
        {
            auto gv = mlir::dyn_cast< mlir::LLVM::GlobalOp >(op->getParentOp());
            // It has a very different conversion outside of global op.
            if (!gv)
                return logical_result::failure();

            // Here we need to build the final value to be returned.
            auto trg_type = this->convert(gv.getType());
            auto value = construct_value(rewriter, ops.getResult().getDefiningOp(), trg_type);

            rewriter.template create< mlir::LLVM::ReturnOp >(
                    op.getLoc(),
                    value);
            rewriter.eraseOp(op);
            return logical_result::success();
        }
    };

    struct sizeof_pattern : base_pattern< hl::SizeOfTypeOp >
    {
        using op_t = hl::SizeOfTypeOp;
        using base = base_pattern< op_t >;
        using base::base;

        using adaptor_t = typename op_t::Adaptor;

        logical_result matchAndRewrite(
            op_t op, adaptor_t ops, conversion_rewriter &rewriter
        ) const override {
            // TODO mimic: clang/lib/CodeGen/CGExprScalar.cpp:VisitUnaryExprOrTypeTraitExpr
            // This does not consider type alignment and VLA types
            auto target_type = this->convert(op.getType());
            const auto &dl = this->type_converter().getDataLayoutAnalysis()->getAtOrAbove(op);
            auto attr = rewriter.getIntegerAttr(
                target_type, dl.getTypeSize(op.getArg())
            );
            auto cons = rewriter.create< LLVM::ConstantOp >(
                op.getLoc(), target_type, attr
            );
            rewriter.replaceOp(op, cons);
            return logical_result::success();
        }
    };

    using base_op_conversions = util::type_list<
        func_op< hl::FuncOp >,
        func_op< ll::FuncOp >,
        constant_int,
        implicit_cast,
        cstyle_cast,
        call,
        icmp,
        fcmp,
        deref,
        subscript_like< ll::Subscript >,
        subscript_like< hl::SubscriptOp >,
        sizeof_pattern,
        propagate_yield< hl::ExprOp, hl::ValueYieldOp >,
        value_yield_in_global_var
    >;

    // Drop types of operations that will be processed by pass for core(lazy) operations.
    template< typename LazyOp >
    struct lazy_op_type : base_pattern< LazyOp >
    {
        using base = base_pattern< LazyOp >;
        using base::base;

        logical_result matchAndRewrite(
            LazyOp op, typename LazyOp::Adaptor ops,
            conversion_rewriter &rewriter) const override
        {
            // It does have regions, we cannot use `clone` as it will screw the
            // queue of ops that needs to be converted conversion to-be done.
            if (op->getNumRegions()) {
                auto lower_res_type = [&]()
                {
                    for (std::size_t i = 0; i < op->getNumResults(); ++i) {
                        auto result = op->getResult(i);
                        result.setType(this->convert(result.getType()));
                    }
                };

                // TODO: Should we use clone instead?
                rewriter.modifyOpInPlace(op, lower_res_type);
                return logical_result::success();
            }

            // It does not have regions
            return this->update_via_clone(rewriter, op, ops.getOperands());
        }
    };

    template< typename yield_like >
    struct fixup_yield_types : base_pattern< yield_like >
    {
        using op_t = yield_like;
        using base = base_pattern< yield_like >;
        using base::base;

        logical_result matchAndRewrite(
            op_t op, typename op_t::Adaptor ops,
            conversion_rewriter &rewriter) const override
        {
            // TODO(conv:irstollvm): If we have more patterns to same op
            //                       that are exclusive, can we have one
            //                       place to "dispatch" them?
            if (mlir::isa< mlir::LLVM::GlobalOp >(op->getParentOp()))
                return logical_result::failure();

            // Some operations need to keep it even with void value.
            if (!mlir::isa< core::LazyOp >(op->getParentOp())) {
                if (ops.getResult().getType().template isa< mlir::LLVM::LLVMVoidType >())
                {
                    rewriter.eraseOp(op);
                    return logical_result::success();
                }
            }

            // TODO: What would it take to make this work `updateRootInPlace`?
            return this->update_via_clone(rewriter, op, ops.getOperands());
        }
    };

    struct module_conversion : base_pattern< core::ModuleOp >
    {
        using base      = base_pattern< core::ModuleOp >;
        using base::base;

        using op_t      = core::ModuleOp;
        using adaptor_t = core::ModuleOp::Adaptor;

        logical_result matchAndRewrite(
            op_t op, adaptor_t adaptor, conversion_rewriter &rewriter
        ) const override {
            auto mod = rewriter.create< mlir::ModuleOp >(op.getLoc(), op.getName());
            rewriter.inlineRegionBefore(op.getBody(), mod.getBody());

            // Remove the terminator block that was automatically added by builder
            rewriter.eraseBlock(&mod.getBodyRegion().back());
            mod->setAttrs(op->getAttrs());
            rewriter.eraseOp(op);
            return mlir::success();
        }
    };

    using lazy_op_type_conversions = util::type_list<
        lazy_op_type< core::LazyOp >,
        lazy_op_type< core::BinLAndOp >,
        lazy_op_type< core::BinLOrOp >,
        lazy_op_type< core::SelectOp >,
        fixup_yield_types< hl::ValueYieldOp >
    >;

    // `ll.` memory operations

    struct ll_load : base_pattern< ll::Load >
    {
        using op_t = ll::Load;
        using base = base_pattern< op_t >;
        using base::base;

        logical_result matchAndRewrite(
            op_t op, typename op_t::Adaptor ops,
            conversion_rewriter &rewriter) const override
        {
            auto trg = convert(op.getResult().getType());
            auto load = rewriter.create< mlir::LLVM::LoadOp >(op.getLoc(), trg, ops.getPtr());

            rewriter.replaceOp(op, load);
            return mlir::success();
        }
    };

    struct ll_store : base_pattern< ll::Store >
    {
        using op_t = ll::Store;
        using base = base_pattern< op_t >;
        using base::base;

        logical_result matchAndRewrite(
            op_t op, typename op_t::Adaptor ops,
            conversion_rewriter &rewriter) const override
        {
            auto store = rewriter.create< LLVM::StoreOp >(
                op.getLoc(), ops.getVal(), ops.getPtr());
            rewriter.replaceOp(op, store);
            return mlir::success();
        }
    };

    struct ll_alloca : base_pattern< ll::Alloca >
    {
        using op_t = ll::Alloca;
        using base = base_pattern< op_t >;
        using base::base;

        logical_result matchAndRewrite(
            op_t op, typename op_t::Adaptor ops,
            conversion_rewriter &rewriter) const override
        {
            auto et = converted_element_type(op.getType());
            auto alloca = mk_alloca(rewriter, convert(op.getType()), et, op.getLoc());
            rewriter.replaceOp(op, alloca);

            return mlir::success();
        }
    };

    using ll_memory_ops = util::type_list<
        ll_load,
        ll_store,
        ll_alloca
    >;

    struct IRsToLLVMPass : LLVMConversionPassMixin< IRsToLLVMPass, IRsToLLVMBase >
    {
        using base = LLVMConversionPassMixin< IRsToLLVMPass, IRsToLLVMBase >;

        static conversion_target create_conversion_target(mcontext_t &context, auto &tc) {
            conversion_target target(context);

            target.addIllegalDialect< hl::HighLevelDialect >();
            target.addIllegalDialect< ll::LowLevelDialect >();
            target.addLegalDialect< core::CoreDialect >();
            target.addLegalDialect< mlir::LLVM::LLVMDialect >();

            auto legal_with_llvm_ret_type = [&]< typename T >( T && )
            {
                auto query = tc->template get_has_legal_return_type< T >();
                target.addDynamicallyLegalOp< T >(std::move(query));
            };

            legal_with_llvm_ret_type( core::LazyOp{} );
            legal_with_llvm_ret_type( core::BinLAndOp{} );
            legal_with_llvm_ret_type( core::BinLOrOp{} );
            legal_with_llvm_ret_type( core::SelectOp{} );

            target.addDynamicallyLegalOp< hl::ValueYieldOp >([&](hl::ValueYieldOp op) {
                return mlir::isa< core::LazyOp >(op->getParentOp()) &&
                       tc->template get_has_legal_operand_types< hl::ValueYieldOp >()(op);
            });

            target.addIllegalOp< mlir::func::FuncOp >();

            auto is_legal = tc->get_is_type_conversion_legal();
            target.markUnknownOpDynamicallyLegal(is_legal);

            return target;
        }

        void run_after_conversion() {
            mcontext_t &mctx = getContext();
            conversion_target target(mctx);

            target.addIllegalOp< core::ModuleOp >();
            target.addLegalOp< mlir::ModuleOp >();

            rewrite_pattern_set patterns(&mctx);
            patterns.add< module_conversion >(*tc);

            if (mlir::failed(mlir::applyPartialConversion(getOperation(), target, std::move(patterns)))) {
                return signalPassFailure();
            }
        }

        static void populate_conversions(auto &cfg) {
            base::populate_conversions<
                one_to_one_conversions,
                shift_conversions,
                inline_region_from_op_conversions,
                return_conversions,
                unary_in_place_conversions,
                sign_conversions,
                init_conversions,
                base_op_conversions,
                ignore_patterns,
                label_patterns,
                lazy_op_type_conversions,
                ll_generic_patterns,
                ll_cf::conversions,
                ll_memory_ops
            >(cfg);
        }

        static void set_lower_to_llvm_options(lower_to_llvm_options &llvm_options) {
            llvm_options.useBarePtrCallConv = true;
        }
    };
} // namespace vast::conv


std::unique_ptr< mlir::Pass > vast::createIRsToLLVMPass()
{
    return std::make_unique< vast::conv::irstollvm::IRsToLLVMPass >();
}
