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
#include "vast/Dialect/Core/Linkage.hpp"

#include "vast/Dialect/LowLevel/LowLevelOps.hpp"

#include "vast/Dialect/Core/CoreOps.hpp"
#include "vast/Dialect/Core/TypeTraits.hpp"

#include "vast/Util/Common.hpp"
#include "vast/Util/Symbols.hpp"
#include "vast/Util/Terminator.hpp"
#include "vast/Util/TypeList.hpp"

#include "vast/Conversion/Common/Mixins.hpp"
#include "vast/Conversion/TypeConverters/LLVMTypeConverter.hpp"

#include "LLCFToLLVM.hpp"

namespace vast::conv::irstollvm
{
    namespace LLVM = mlir::LLVM;

    tc::lower_to_llvm_options mk_default_opts(mcontext_t *mctx) {
        tc::lower_to_llvm_options opts(mctx);
        opts.useBarePtrCallConv = true;
        return opts;
    }

    template< typename op_t >
    struct llvm_conversion_pattern
        : operation_conversion_pattern< op_t >
        , llvm_pattern_utils
    {
        using base = operation_conversion_pattern< op_t >;
        using base::base;

        // const mlir::DataLayoutAnalysis &dl;

        tc::llvm_type_converter tc(operation from) const {
            auto dl = mlir::DataLayoutAnalysis(from);
            tc::lower_to_llvm_options opts(from->getContext(), dl.getAtOrAbove(from));
            return tc::llvm_type_converter(from->getContext(), dl, opts, from);
        }

        mlir_type convert_type_to_type(operation from, mlir_type type) const {
            auto t = tc(from);
            return t.convert_type_to_type(type).value();
        }

        mlir_type convert_element_type(operation from, mlir_type type) const {
            return tc(from).convert_type_to_type(
                mlir::cast< ElementTypeInterface >(type).getElementType()
            ).value();
        }

        auto mk_index(auto loc, std::size_t idx, auto &rewriter) const
            -> mlir::LLVM::ConstantOp
        {
            mlir::LLVMTypeConverter tc(rewriter.getContext());
            auto index_type = tc.convertType(rewriter.getIndexType());
            return rewriter.template create< mlir::LLVM::ConstantOp >(
                loc, index_type, rewriter.getIntegerAttr(index_type, idx)
            );
        }
    };

    template< typename op_t >
    struct non_legalizing_llvm_conversion_pattern : llvm_conversion_pattern< op_t >
    {
        using base = llvm_conversion_pattern< op_t >;
        using base::base;
        using adaptor_t = typename op_t::Adaptor;

        static void legalize(conversion_target &) {}
    };

    template< typename src_t, typename trg_t >
    struct llvm_one_to_one_conversion_pattern : llvm_conversion_pattern< src_t >
    {
        using base = llvm_conversion_pattern< src_t >;
        using base::base;
        using adaptor_t = typename src_t::Adaptor;

        logical_result matchAndRewrite(
            src_t op, adaptor_t ops, conversion_rewriter &rewriter
        ) const override {
            auto target_ty = this->convert_type_to_type(op, op.getType());
            auto new_op = rewriter.create< trg_t >(op.getLoc(), target_ty, ops.getOperands());
            rewriter.replaceOp(op, new_op);
            return mlir::success();
        }

        static void legalize(conversion_target &target) {
            target.addIllegalOp< src_t >();
            target.addLegalOp< trg_t >();
        }
    };

    logical_result update_via_clone(auto &rewriter, auto op, auto operands, auto &tc) {
        auto new_op = rewriter.clone(*op);
        new_op->setOperands(operands);
        for (auto v : new_op->getResults()) {
            v.setType(tc.convert_type_to_type(v.getType()).value());
        }
        rewriter.replaceOp(op, new_op);
        return mlir::success();
    }

    using operands_forwarding_patterns = util::type_list<
        operands_forwarding_pattern< hl::PredefinedExpr >,
        operands_forwarding_pattern< hl::AddressOf >,
        operands_forwarding_pattern< hl::NullStmt >
    >;

    using erase_patterns = util::type_list<
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
            auto trg_type = self().convert_type_to_type(op, get_only_result(op).getType());
            return construct_value(rewriter, op, trg_type);
        }

        mlir_value construct_value(auto &rewriter, operation op, mlir_type type) const {
            auto init_list = mlir::dyn_cast< hl::InitListExpr >(op);
            if (!init_list) {
                return get_only_result(op);
            }
            auto trg_type = self().convert_type_to_type(op, type);
            return construct_value(rewriter, init_list.getLoc(), init_list.getElements(), trg_type);
        }

        mlir_value construct_value(
            auto &rewriter, auto loc, const auto &operands, mlir_type trg_type
        ) const {
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
            for (auto [idx, element] : llvm::enumerate(operands)) {
                auto elem = construct_value(rewriter, element);
                init = rewriter.template create< mlir::LLVM::InsertValueOp >(
                    loc, init, elem, idx
                );
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

        mlir_value undef(auto &rewriter, auto loc, auto type) const {
            return rewriter.template create< mlir::LLVM::UndefOp >(loc, type);
        }
    };

    struct ll_struct_gep : llvm_conversion_pattern< ll::StructGEPOp >
    {
        using op_t = ll::StructGEPOp;
        using base = llvm_conversion_pattern< op_t >;
        using adaptor_t = typename op_t::Adaptor;

        using base::base;

        logical_result matchAndRewrite(
            op_t op, adaptor_t ops, conversion_rewriter &rewriter
        ) const override {
            std::vector< mlir::LLVM::GEPArg > indices{ 0ul, ops.getIdx() };

            auto ptr = mlir::dyn_cast< hl::PointerType >(op.getRecord().getType());
            VAST_CHECK(ptr, "{0} is not a pointer to record!", op.getRecord().getType());

            auto tc = this->tc(op);

            auto gep = rewriter.create< mlir::LLVM::GEPOp >(
                op.getLoc(),
                tc.convert_type_to_type(op.getType()).value(),
                tc.convert_type_to_type(ptr.getElementType()).value(),
                ops.getRecord(),
                indices
            );

            rewriter.replaceOp(op, gep);
            return mlir::success();
        }
    };

    struct ll_extract : llvm_conversion_pattern< ll::Extract >
    {
        using op_t = ll::Extract;
        using base = llvm_conversion_pattern< op_t >;
        using adaptor_t = typename op_t::Adaptor;

        using base::base;

        logical_result matchAndRewrite(
            op_t op, adaptor_t ops, conversion_rewriter &rewriter
        ) const override {
            auto loc = op.getLoc();

            auto value = ops.getArg();
            auto from = op.from();

            auto shift = rewriter.create< mlir::LLVM::LShrOp >(
                loc, value, iN(rewriter, loc, value.getType(), from)
            );

            auto trg_type = convert_type_to_type(op, op.getType());
            auto trunc = rewriter.create< mlir::LLVM::TruncOp >(loc, trg_type, shift);
            rewriter.replaceOp(op, trunc);
            return mlir::success();
        }
    };

    struct ll_concat : llvm_conversion_pattern< ll::Concat >
    {
        using op_t = ll::Concat;
        using base = llvm_conversion_pattern< op_t >;
        using adaptor_t = typename op_t::Adaptor;

        using base::base;

        std::size_t bw(operation op) const {
            VAST_ASSERT(op->getNumResults() == 1);
            auto dl = mlir::DataLayoutAnalysis(op);
            return dl.getAtOrAbove(op).getTypeSizeInBits(
                convert_type_to_type(op, op->getResult(0).getType())
            );
        }

        logical_result matchAndRewrite(
            op_t op, adaptor_t ops, conversion_rewriter &rewriter
        ) const override {
            auto loc = op.getLoc();

            auto resize = [&](auto w) -> mlir_value {
                auto trg_type = convert_type_to_type(op, op.getType());
                if (w.getType() == trg_type) {
                    return w;
                }
                return rewriter.create< mlir::LLVM::ZExtOp >(loc, trg_type, w);
            };
            mlir_value head = resize(ops.getOperands()[0]);

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
    struct inline_region_from_op : operation_conversion_pattern< Op >
    {
        using base = operation_conversion_pattern< Op >;
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

    using label_stmt = cf::region_to_block_conversion_pattern< hl::LabelStmt >;
    using scope_op   = cf::region_to_block_conversion_pattern< core::ScopeOp >;

    using label_patterns = util::type_list< erase_pattern< hl::LabelDeclOp >, label_stmt >;

    // TODO(conv): Figure out if these can be somehow unified.
    using inline_region_from_op_conversions = util::type_list<
        inline_region_from_op< hl::TranslationUnitOp >, scope_op
    >;

    template< typename op_t >
    struct subscript_like : llvm_conversion_pattern< op_t >
    {
        using base = llvm_conversion_pattern< op_t >;
        using adaptor_t = typename op_t::Adaptor;

        using base::base;

        logical_result matchAndRewrite(
            op_t op, adaptor_t ops, conversion_rewriter &rewriter
        ) const override {
            auto trg_type = this->convert_type_to_type(op, op.getType());
            auto element_type = this->convert_element_type(op, op.getType());

            VAST_PATTERN_CHECK(trg_type, "Could not convert vardecl type");

            auto gep = rewriter.create< mlir::LLVM::GEPOp >(
                op.getLoc(),
                trg_type, element_type,
                ops.getArray(), ops.getIndex()
            );

            rewriter.replaceOp(op, gep);
            return logical_result::success();
        }
    };

    struct init_list_expr
        : llvm_conversion_pattern< hl::InitListExpr >
        , value_builder< init_list_expr >
    {
        using op_t = hl::InitListExpr;
        using base = llvm_conversion_pattern< op_t >;
        using adaptor_t = typename op_t::Adaptor;

        using base::base;

        logical_result matchAndRewrite(
            op_t op, adaptor_t ops, conversion_rewriter &rewriter
        ) const override {
            VAST_PATTERN_CHECK(op.getNumResults() == 1, "Unexpected number of results");
            auto trg_type = convert_type_to_type(op, op.getType(0));
            auto value = construct_value(
                rewriter, op.getLoc(), ops.getOperands(), trg_type
            );

            rewriter.replaceOp(op, value);
            return mlir::success();
        }
    };

    struct vardecl : llvm_conversion_pattern< hl::VarDeclOp >
    {
        using op_t = hl::VarDeclOp;
        using base = llvm_conversion_pattern< op_t >;
        using adaptor_t = typename op_t::Adaptor;

        using base::base;

        logical_result matchAndRewrite(
            op_t op, adaptor_t ops, conversion_rewriter &rewriter
        ) const override {
            auto t = mlir::dyn_cast< hl::PointerType >(op.getType());
            auto target_type = convert_type_to_type(op, t.getElementType());

            // So we know this is a global, otherwise it would be in `ll:`.
            auto linkage = op.getLinkage();
            if (!linkage) {
                if (op.isStaticLocal()) {
                    linkage = core::GlobalLinkageKind::InternalLinkage;
                } else {
                    VAST_REPORT("Global var without linkage information.");
                    return mlir::failure();
                }
            }

            auto gop = rewriter.create< mlir::LLVM::GlobalOp >(
                    op.getLoc(),
                    target_type,
                    op.getConstant(),
                    core::convert_linkage_to_llvm(linkage.value()),
                    op.getSymbolName(),
                    mlir::Attribute()
            );

            // We could probably try to analyze the region to see if it isn't
            // a case where we can just do an attribute, but for now let's
            // just use the initializer.
            auto &region = gop.getInitializerRegion();
            rewriter.inlineRegionBefore(op.getInitializer(),
                                        region, region.begin());

            auto &gop_init = gop.getInitializer();
            if (gop_init.empty() && op.getStorageDuration() == core::StorageDuration::sd_static) {
                auto guard = insertion_guard(rewriter);
                auto &init_block = gop_init.emplaceBlock();
                rewriter.setInsertionPoint(&init_block, init_block.begin());
                auto zero_init = rewriter.create< mlir::LLVM::ZeroOp >(op.getLoc(), gop.getType());
                rewriter.create< mlir::LLVM::ReturnOp >(op.getLoc(), zero_init);
            }
            rewriter.eraseOp(op);
            return logical_result::success();
        }

    };

    struct global_ref : llvm_conversion_pattern< hl::DeclRefOp >
    {
        using op_t = hl::DeclRefOp;
        using base = llvm_conversion_pattern< op_t >;
        using adaptor_t = typename op_t::Adaptor;

        using base::base;

        logical_result matchAndRewrite(
            op_t op, adaptor_t ops, conversion_rewriter &rewriter
        ) const override {
            auto target_type = convert_type_to_type(op, op.getType());

            auto addr_of = rewriter.template create< mlir::LLVM::AddressOfOp >(
                op.getLoc(),
                target_type,
                op.getName()
            );

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
    struct func_op : llvm_conversion_pattern< op_t >
    {
        using base = llvm_conversion_pattern< op_t >;
        using adaptor_t = typename op_t::Adaptor;

        using base::base;

        using llvm_func_op = mlir::LLVM::LLVMFuncOp;

        logical_result matchAndRewrite(
            op_t func_op, adaptor_t ops, conversion_rewriter &rewriter
        ) const override {
            auto tc = this->tc(func_op);
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


            auto linkage = func_op.getLinkage();
            VAST_CHECK(linkage, "Attempting lower function without set linkage {0}", func_op);
            auto new_func = rewriter.create< llvm_func_op >(
                func_op.getLoc(),
                func_op.getSymbolName(),
                target_type,
                core::convert_linkage_to_llvm(linkage.value()),
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

    struct constant : llvm_conversion_pattern< hl::ConstantOp >
    {
        using op_t = hl::ConstantOp;
        using base = llvm_conversion_pattern< op_t >;
        using adaptor_t = typename op_t::Adaptor;

        using base::base;

        static inline constexpr const char *strlit_global_var_prefix = "vast.strlit.constant_";

        std::string next_strlit_name(core::module mod) const {
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

        logical_result handle_void_const(op_t op, conversion_rewriter &rewriter) const {
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
            op_t op, adaptor_t ops, conversion_rewriter &rewriter
        ) const override {
            if (mlir::isa< mlir::NoneType >(op.getResult().getType())) {
                return handle_void_const(op, rewriter);
            }

            auto val = make_from(op, rewriter, this->tc(op));
            if (!val)
                return mlir::failure();

            rewriter.replaceOp(op, val);
            return logical_result::success();
        }

        mlir::Attribute convert_attr(
            auto attr, auto op, conversion_rewriter &rewriter
        ) const {
            auto target_type = convert_type_to_type(op, attr.getType());
            auto dla = mlir::DataLayoutAnalysis(op);
            const auto &dl = dla.getAtOrAbove(op);
            if (!target_type)
                return {};

            if (auto float_attr = mlir::dyn_cast< core::FloatAttr >(attr)) {
                // NOTE(lukas): We cannot simply forward the return value of `getValue()`
                //              because it can have different semantics than one expected
                //              by `FloatAttr`.
                // TODO(lukas): Is there a better way to convert this?
                //              Ideally `APFloat -> APFloat`.
                double raw_value = float_attr.getValue().convertToDouble();
                return rewriter.getFloatAttr(target_type, raw_value);
            }

            if (auto int_attr = mlir::dyn_cast< core::IntegerAttr >(attr)) {
                auto size = dl.getTypeSizeInBits(target_type);
                auto coerced = int_attr.getValue().sextOrTrunc(size);
                return rewriter.getIntegerAttr(target_type, coerced);
            }

            VAST_UNREACHABLE("Trying to convert attr that is not supported, {0} in op {1}",
                attr, op
            );

            return {};
        }

        mlir_value convert_strlit(
            op_t op, auto rewriter, auto &&tc, mlir_type original_type, mlir::StringAttr str_lit
        ) const {
            // We need to include the terminating `0` which will not happen
            // if we "just" pass the value in.
            auto twine = llvm::Twine(std::string_view(
                str_lit.getValue().data(), str_lit.getValue().size() + 1)
            );

            auto converted_attr = mlir::StringAttr::get(op->getContext(), twine);

            auto mod = op->getParentOfType< core::module >();
            auto name = next_strlit_name(mod);

            auto element_type = this->convert_element_type(op, op.getType());
            rewriter.guarded([&]() {
                rewriter->setInsertionPoint(&*mod.begin());
                rewriter->template create< mlir::LLVM::GlobalOp >(
                    op.getLoc(),
                    element_type,
                    true, /* is constant */
                    LLVM::Linkage::Internal,
                    name,
                    converted_attr
                );
            });

            return rewriter->template create< mlir::LLVM::AddressOfOp >(
                op.getLoc(), tc.convert_type_to_type(original_type).value(), name
            );
        }

        mlir_value make_from(op_t op, conversion_rewriter &rewriter, auto &&tc) const {
            auto target_type = convert_type_to_type(op, op.getType());

            if (auto str_lit = mlir::dyn_cast< mlir::StringAttr >(op.getValue())) {
                return convert_strlit(
                    op, rewriter_wrapper_t(rewriter), tc, op.getType(), str_lit
                );
            }

            auto attr = convert_attr(op.getValue(), op, rewriter);
            if (!attr) {
                return {};
            }

            return rewriter.create< LLVM::ConstantOp >(op.getLoc(), target_type, attr);
        }
    };

    template< typename op_t >
    struct ret : operation_conversion_pattern< op_t >
    {
        using base = operation_conversion_pattern< op_t >;
        using adaptor_t = typename op_t::Adaptor;

        using base::base;

        std::vector< mlir_value > filter_out_void(const auto &values) const {
            std::vector< mlir_value > out;
            for (auto v : values) {
                auto ty = v.getType();
                if (!mlir::isa< mlir::NoneType >(ty) && !mlir::isa< LLVM::LLVMVoidType >(ty)) {
                    out.push_back(v);
                }
            }
            return out;
        }

        logical_result matchAndRewrite(
            op_t ret_op, adaptor_t ops, conversion_rewriter &rewriter
        ) const override{
            rewriter.create< LLVM::ReturnOp >(
                ret_op.getLoc(), filter_out_void(ops.getOperands())
            );
            rewriter.eraseOp(ret_op);
            return logical_result::success();
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
        auto dst_type = pattern.convert_type_to_type(op, op.getType());

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
            auto et = pattern.convert_element_type(op, op.getType());
            rewriter.template replaceOpWithNewOp< LLVM::GEPOp >(
                op, dst_type, et, src, LLVM::GEPArg{ 0 }
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

    struct implicit_cast : llvm_conversion_pattern< hl::ImplicitCastOp >
    {
        using op_t = hl::ImplicitCastOp;
        using base = llvm_conversion_pattern< hl::ImplicitCastOp >;
        using adaptor_t = typename op_t::Adaptor;

        using base::base;

        logical_result matchAndRewrite(
            op_t op, adaptor_t adaptor, conversion_rewriter &rewriter
        ) const override {
            return match_and_rewrite_cast(*this, op, adaptor, rewriter);
        }
    };

    struct cstyle_cast
        : llvm_conversion_pattern< hl::CStyleCastOp >
    {
        using op_t = hl::CStyleCastOp;
        using base = llvm_conversion_pattern< op_t >;
        using adaptor_t = typename op_t::Adaptor;

        using base::base;

        logical_result matchAndRewrite(
            op_t op, adaptor_t adaptor, conversion_rewriter &rewriter
        ) const override {
            return match_and_rewrite_cast(*this, op, adaptor, rewriter);
        }
    };

    template< typename src_t, typename trg_t >
    struct shift_op : llvm_conversion_pattern< src_t >
    {
        using base = llvm_conversion_pattern< src_t >;
        using adaptor_t = typename src_t::Adaptor;
        using base::base;

        logical_result matchAndRewrite(
            src_t op, adaptor_t ops, conversion_rewriter &rewriter
        ) const override {
            auto target_ty = this->convert_type_to_type(op, op.getType());
            auto rhs       = base::insert_trunc_or_ext(ops.getRhs(), target_ty, rewriter);
            auto new_op = rewriter.create< trg_t >(op.getLoc(), target_ty, ops.getLhs(), rhs);
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
        llvm_one_to_one_conversion_pattern< hl::AddIOp, LLVM::AddOp >,
        llvm_one_to_one_conversion_pattern< hl::SubIOp, LLVM::SubOp >,
        llvm_one_to_one_conversion_pattern< hl::MulIOp, LLVM::MulOp >,

        llvm_one_to_one_conversion_pattern< hl::AddFOp, LLVM::FAddOp >,
        llvm_one_to_one_conversion_pattern< hl::SubFOp, LLVM::FSubOp >,
        llvm_one_to_one_conversion_pattern< hl::MulFOp, LLVM::FMulOp >,

        llvm_one_to_one_conversion_pattern< hl::DivSOp, LLVM::SDivOp >,
        llvm_one_to_one_conversion_pattern< hl::DivUOp, LLVM::UDivOp >,
        llvm_one_to_one_conversion_pattern< hl::DivFOp, LLVM::FDivOp >,

        llvm_one_to_one_conversion_pattern< hl::RemSOp, LLVM::SRemOp >,
        llvm_one_to_one_conversion_pattern< hl::RemUOp, LLVM::URemOp >,
        llvm_one_to_one_conversion_pattern< hl::RemFOp, LLVM::FRemOp >,

        llvm_one_to_one_conversion_pattern< hl::BinOrOp, LLVM::OrOp >,
        llvm_one_to_one_conversion_pattern< hl::BinAndOp, LLVM::AndOp >,
        llvm_one_to_one_conversion_pattern< hl::BinXorOp, LLVM::XOrOp >
    >;

    struct call : llvm_conversion_pattern< hl::CallOp >
    {
        using op_t = hl::CallOp;
        using base = llvm_conversion_pattern< op_t >;
        using adaptor_t = typename op_t::Adaptor;

        using base::base;

        logical_result matchAndRewrite(
            op_t op, adaptor_t ops, conversion_rewriter &rewriter
        ) const override {
            auto caller = mlir::dyn_cast< VastCallOpInterface >(op.getOperation());
            if (!caller) {
                return logical_result::failure();
            }

            auto callee = caller.resolveCallable();
            if (!callee && !mlir::isa< core::function_op_interface >(callee)) {
                return logical_result::failure();
            }

            auto fn   = mlir::cast< core::function_op_interface >(callee);
            auto fty  = mlir::cast< core::FunctionType >(fn.getFunctionType());
            auto rtys = this->tc(op).convert_types_to_types(fn.getResultTypes());
            auto atys = this->tc(op).convert_types_to_types(fn.getArgumentTypes());

            if (!rtys) {
                return logical_result::failure();
            }

            auto mk_fty = [&] {
                mlir_type rty = rtys->empty()
                    ? mlir::LLVM::LLVMVoidType::get(op.getContext())
                    : rtys->front();
                return mlir::LLVM::LLVMFunctionType::get(rty, atys.value(), fty.isVarArg());
            };

            auto call = rewriter.create< mlir::LLVM::CallOp >(
                op.getLoc(), mk_fty(), op.getCallee(), ops.getOperands()
            );

            // The result gets removed when return type is void,
            // because the number of results is mismatched, we can't use replace (triggers assert)
            // Removing the op is ok, since in llvm dialect a void value can't be used anyway
            if (call.getResult())
                rewriter.replaceOp(op, call.getResults());
            else
                rewriter.eraseOp(op);

            return logical_result::success();
        }
    };

    struct logical_not : llvm_conversion_pattern< hl::LNotOp >
    {
        using op_t = hl::LNotOp;
        using base = llvm_conversion_pattern< op_t >;
        using adaptor_t = typename op_t::Adaptor;

        using base::base;

        Operation * create_cmp(conversion_rewriter &rewriter, op_t op, Value lhs, Value rhs) const {
            auto lhs_type = lhs.getType();

            if (llvm::isa< mlir::FloatType >(lhs_type)) {
                auto i1 = mlir::IntegerType::get(
                    op.getContext(), 1, mlir::IntegerType::Signless
                );

                return rewriter.create< LLVM::FCmpOp >(
                    op.getLoc(), i1,
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

        logical_result matchAndRewrite(
            op_t op, adaptor_t adaptor, conversion_rewriter &rewriter
        ) const override {
            auto zero = this->constant(rewriter, op.getLoc(), adaptor.getArg().getType(), 0);

            auto cmp = create_cmp(rewriter, op, adaptor.getArg(), zero);

            auto true_i1 = this->iN(rewriter, op.getLoc(), cmp->getResult(0).getType(), 1);
            auto xor_val = rewriter.create< LLVM::XOrOp >(op.getLoc(), cmp->getResult(0), true_i1);

            auto res_type = convert_type_to_type(op, op.getResult().getType());
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

    struct bin_not : operation_conversion_pattern< hl::NotOp >, llvm_pattern_utils
    {
        using op_t = hl::NotOp;
        using base = operation_conversion_pattern< op_t >;
        using adaptor_t = typename op_t::Adaptor;

        using base::base;

        logical_result matchAndRewrite(
            op_t op, adaptor_t adaptor, conversion_rewriter &rewriter
        ) const override {
            auto helper = this->constant(rewriter, op.getLoc(), adaptor.getArg().getType(), -1);
            rewriter.replaceOpWithNewOp< LLVM::XOrOp >(op, adaptor.getArg(), helper);
            return logical_result::success();
        }
    };

    using unary_in_place_conversions = util::type_list<
        logical_not,
        bin_not
    >;

    struct minus : llvm_conversion_pattern< hl::MinusOp >
    {
        using op_t = hl::MinusOp;
        using base = llvm_conversion_pattern< op_t >;
        using adaptor_t = typename op_t::Adaptor;

        using base::base;

        logical_result matchAndRewrite(
            op_t op, adaptor_t adaptor, conversion_rewriter &rewriter
        ) const override {
            auto arg = adaptor.getArg();
            auto arg_type = convert_type_to_type(op, arg.getType());

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
        operands_forwarding_pattern< hl::PlusOp >
    >;

    template< typename src_t, typename trg_t, typename src_predicate_t, typename trg_predicate_t >
    struct cmp_base : llvm_conversion_pattern< src_t >
    {
        using base = llvm_conversion_pattern< src_t >;
        using adaptor_t = typename src_t::Adaptor;

        using base::base;

        virtual trg_predicate_t convert_predicate(src_predicate_t predicate) const = 0;

        logical_result matchAndRewrite(
            src_t op, adaptor_t adaptor, conversion_rewriter &rewriter
        ) const override {
            auto pred = convert_predicate(op.getPredicate());

            auto new_cmp = rewriter.create< trg_t >(
                op.getLoc(), pred, adaptor.getLhs(), adaptor.getRhs()
            );

            auto dst_type = this->convert_type_to_type(op, op.getType());
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

    struct deref : llvm_conversion_pattern< hl::Deref >
    {
        using op_t = hl::Deref;
        using base = llvm_conversion_pattern< hl::Deref >;
        using adaptor_t = typename op_t::Adaptor;

        using base::base;

        logical_result matchAndRewrite(
            op_t op, adaptor_t ops, conversion_rewriter &rewriter
        ) const override {
            auto trg_type = convert_type_to_type(op, op.getType());
            auto loaded = rewriter.create< mlir::LLVM::LoadOp >(
                op.getLoc(), trg_type, ops.getAddr()
            );

            rewriter.replaceOp(op, loaded);
            return logical_result::success();
        }
    };

    template< typename op_t, typename yield_op_t >
    struct propagate_yield : non_legalizing_llvm_conversion_pattern< op_t >
    {
        using base = non_legalizing_llvm_conversion_pattern< op_t >;
        using adaptor_t = typename op_t::Adaptor;

        using base::base;

        logical_result matchAndRewrite(
            op_t op, adaptor_t ops, conversion_rewriter &rewriter
        ) const override {
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

    struct value_yield_in_global_var
        : non_legalizing_llvm_conversion_pattern< hl::ValueYieldOp >
        , value_builder< value_yield_in_global_var >
    {
        using op_t = hl::ValueYieldOp;
        using base = non_legalizing_llvm_conversion_pattern< op_t >;
        using adaptor_t = typename op_t::Adaptor;

        using base::base;

        logical_result matchAndRewrite(
            op_t op, adaptor_t ops, conversion_rewriter &rewriter) const override
        {
            auto gv = mlir::dyn_cast< mlir::LLVM::GlobalOp >(op->getParentOp());
            // It has a very different conversion outside of global op.
            if (!gv)
                return logical_result::failure();

            // Here we need to build the final value to be returned.
            auto trg_type = convert_type_to_type(op, gv.getType());
            auto value = construct_value(rewriter, ops.getResult().getDefiningOp(), trg_type);

            rewriter.template create< mlir::LLVM::ReturnOp >(op.getLoc(), value);
            rewriter.eraseOp(op);
            return logical_result::success();
        }
    };

    struct sizeof_pattern : llvm_conversion_pattern< hl::SizeOfTypeOp >
    {
        using op_t = hl::SizeOfTypeOp;
        using base = llvm_conversion_pattern< op_t >;
        using adaptor_t = typename op_t::Adaptor;

        using base::base;

        logical_result matchAndRewrite(
            op_t op, adaptor_t ops, conversion_rewriter &rewriter
        ) const override {
            // TODO mimic: clang/lib/CodeGen/CGExprScalar.cpp:VisitUnaryExprOrTypeTraitExpr
            // This does not consider type alignment and VLA types
            auto target_type = convert_type_to_type(op, op.getType());
            auto dla = mlir::DataLayoutAnalysis(op);
            const auto &dl = dla.getAtOrAbove(op);
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
        constant,
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
    struct lazy_op_type : non_legalizing_llvm_conversion_pattern< LazyOp >
    {
        using op_t = LazyOp;
        using base = non_legalizing_llvm_conversion_pattern< op_t >;
        using adaptor_t = typename LazyOp::Adaptor;
        using base::base;

        logical_result matchAndRewrite(
            op_t op, adaptor_t ops, conversion_rewriter &rewriter
        ) const override {
            // It does have regions, we cannot use `clone` as it will screw the
            // queue of ops that needs to be converted conversion to-be done.
            if (op->getNumRegions()) {
                auto lower_res_type = [&]()
                {
                    for (std::size_t i = 0; i < op->getNumResults(); ++i) {
                        auto result = op->getResult(i);
                        result.setType(this->convert_type_to_type(op, result.getType()));
                    }
                };

                // TODO: Should we use clone instead?
                rewriter.modifyOpInPlace(op, lower_res_type);
                return logical_result::success();
            }

            // It does not have regions
            auto tc = this->tc(op);
            return update_via_clone(rewriter, op, ops.getOperands(), tc);
        }
    };

    template< typename yield_like >
    struct fixup_yield_types : non_legalizing_llvm_conversion_pattern< yield_like >
    {
        using op_t = yield_like;
        using base = non_legalizing_llvm_conversion_pattern< yield_like >;
        using adaptor_t = typename op_t::Adaptor;

        using base::base;

        logical_result matchAndRewrite(
            op_t op, adaptor_t ops, conversion_rewriter &rewriter
        ) const override {
            // TODO(conv:irstollvm): If we have more patterns to same op
            //                       that are exclusive, can we have one
            //                       place to "dispatch" them?
            if (mlir::isa< mlir::LLVM::GlobalOp >(op->getParentOp()))
                return logical_result::failure();

            // Some operations need to keep it even with void value.
            if (!mlir::isa< core::LazyOp >(op->getParentOp())) {
                auto ty = ops.getResult().getType();
                if (mlir::isa< LLVM::LLVMVoidType >(ty)) {
                    rewriter.eraseOp(op);
                    return logical_result::success();
                }
            }

            if (mlir::isa< mlir::NoneType >(ops.getResult().getType())) {
                rewriter.eraseOp(op);
                return logical_result::success();
            }

            // TODO: What would it take to make this work `updateRootInPlace`?
            auto tc = this->tc(op);
            return update_via_clone(rewriter, op, ops.getOperands(),tc);
        }
    };

    struct module_conversion : operation_conversion_pattern< core::ModuleOp >
    {
        using op_t      = core::ModuleOp;
        using base      = operation_conversion_pattern< op_t >;
        using adaptor_t = core::ModuleOp::Adaptor;

        using base::base;

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

    struct ll_load : llvm_conversion_pattern< ll::Load >
    {
        using op_t = ll::Load;
        using base = llvm_conversion_pattern< op_t >;
        using adaptor_t = typename op_t::Adaptor;

        using base::base;

        logical_result matchAndRewrite(
            op_t op, adaptor_t ops, conversion_rewriter &rewriter
        ) const override {
            auto trg = convert_type_to_type(op, op.getResult().getType());
            auto load = rewriter.create< mlir::LLVM::LoadOp >(op.getLoc(), trg, ops.getPtr());

            rewriter.replaceOp(op, load);
            return mlir::success();
        }
    };

    struct ll_store : operation_conversion_pattern< ll::Store >
    {
        using op_t = ll::Store;
        using base = operation_conversion_pattern< op_t >;
        using adaptor_t = typename op_t::Adaptor;

        using base::base;

        logical_result matchAndRewrite(
            op_t op, adaptor_t ops, conversion_rewriter &rewriter
        ) const override {
            auto store = rewriter.create< LLVM::StoreOp >(
                op.getLoc(), ops.getVal(), ops.getPtr()
            );
            rewriter.replaceOp(op, store);
            return mlir::success();
        }
    };

    struct ll_alloca : llvm_conversion_pattern< ll::Alloca >
    {
        using op_t = ll::Alloca;
        using base = llvm_conversion_pattern< op_t >;
        using adaptor_t = typename op_t::Adaptor;

        using base::base;

        logical_result matchAndRewrite(
            op_t op, adaptor_t ops, conversion_rewriter &rewriter
        ) const override {
            auto ty = convert_type_to_type(op, op.getType());
            auto et = convert_element_type(op, op.getType());
            auto count = mk_index(op.getLoc(), 1, rewriter);
            auto alloca = rewriter.template create< LLVM::AllocaOp >(
                op.getLoc(), ty, et, count, 0
            );
            rewriter.replaceOp(op, alloca);

            return mlir::success();
        }
    };

    using ll_memory_ops = util::type_list< ll_load, ll_store, ll_alloca >;

    struct IRsToLLVMPass : ConversionPassMixin< IRsToLLVMPass, IRsToLLVMBase >
    {
        using base = ConversionPassMixin< IRsToLLVMPass, IRsToLLVMBase >;

        static conversion_target create_conversion_target(mcontext_t &mctx) {
            conversion_target target(mctx);

            target.addIllegalDialect< hl::HighLevelDialect >();
            target.addIllegalDialect< ll::LowLevelDialect >();
            target.addLegalDialect< core::CoreDialect >();
            target.addLegalDialect< mlir::LLVM::LLVMDialect >();

            auto has_legal_return_type = [](auto op) {
                auto dla = mlir::DataLayoutAnalysis(op);
                auto opts = mk_default_opts(op->getContext());
                return tc::llvm_type_converter(op->getContext(), dla, opts, op).has_legal_return_type(op);
            };

            auto has_legal_operand_types = [](auto op) {
                auto dla = mlir::DataLayoutAnalysis(op);
                auto opts = mk_default_opts(op->getContext());
                return tc::llvm_type_converter(op->getContext(), dla, opts, op).has_legal_operand_types(op);
            };

            target.addDynamicallyLegalOp< core::LazyOp    >(has_legal_return_type);
            target.addDynamicallyLegalOp< core::BinLAndOp >(has_legal_return_type);
            target.addDynamicallyLegalOp< core::BinLOrOp  >(has_legal_return_type);
            target.addDynamicallyLegalOp< core::SelectOp  >(has_legal_return_type);

            target.addDynamicallyLegalOp< hl::ValueYieldOp >([&](hl::ValueYieldOp op) {
                return mlir::isa< core::LazyOp >(op->getParentOp()) && has_legal_operand_types(op);
            });

            target.addIllegalOp< mlir::func::FuncOp >();

            target.markUnknownOpDynamicallyLegal([&] (auto op) {
                auto dla = mlir::DataLayoutAnalysis(op);
                auto opts = mk_default_opts(&mctx);
                tc::llvm_type_converter tc(op->getContext(), dla, opts, op);
                return tc.get_is_type_conversion_legal()(op);
            });

            return target;
        }

        void run_after_conversion() {
            mcontext_t &mctx = getContext();
            conversion_target target(mctx);

            target.addIllegalOp< core::ModuleOp >();
            target.addLegalOp< mlir::ModuleOp >();

            rewrite_pattern_set patterns(&mctx);
            patterns.add< module_conversion >(&mctx);

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
                operands_forwarding_patterns,
                erase_patterns,
                label_patterns,
                lazy_op_type_conversions,
                ll_generic_patterns,
                cf::patterns,
                ll_memory_ops
            >(cfg);
        }
    };
} // namespace vast::conv


std::unique_ptr< mlir::Pass > vast::createIRsToLLVMPass()
{
    return std::make_unique< vast::conv::irstollvm::IRsToLLVMPass >();
}
