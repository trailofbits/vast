// Copyright (c) 2024-present, Trail of Bits, Inc.

#include "vast/Conversion/Passes.hpp"

VAST_RELAX_WARNINGS
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/BuiltinDialect.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Rewrite/FrozenRewritePatternSet.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
VAST_UNRELAX_WARNINGS

#include "../PassesDetails.hpp"

#include "vast/Dialect/HighLevel/HighLevelTypes.hpp"
#include "vast/Dialect/LowLevel/LowLevelOps.hpp"

#include "vast/Util/Common.hpp"
#include "vast/Util/Terminator.hpp"

#include "vast/Conversion/Common/Mixins.hpp"
#include "vast/Conversion/TypeConverters/TypeConvertingPattern.hpp"

namespace vast::conv {
    namespace {
        struct value_category_type_converter
            : tc::identity_type_converter
            , tc::mixins< value_category_type_converter >
            , tc::function_type_converter< value_category_type_converter >
        {
            mlir::MLIRContext &mctx;

            // Do we need data layout? Probably not as everything should be explicit
            // type size wise.
            value_category_type_converter(mcontext_t &mctx) : mctx(mctx) {
                addConversion([&](hl::LValueType type) {
                    // It should never happen that we have nested lvalues?
                    auto element_type = this->convert_type_to_type(type.getElementType());
                    return hl::PointerType::get(&mctx, *element_type);
                });
                addTargetMaterialization(
                    [&](mlir::OpBuilder &builder, mlir::Type resultType,
                        mlir::ValueRange inputs, mlir::Location loc) -> std::optional< Value > {
                        if (inputs.size() != 1) {
                            return std::nullopt;
                        }

                        return builder
                            .create< mlir::UnrealizedConversionCastOp >(loc, resultType, inputs)
                            .getResult(0);
                    }
                );
                addSourceMaterialization(
                    [&](mlir::OpBuilder &builder, mlir::Type resultType,
                        mlir::ValueRange inputs, mlir::Location loc) -> std::optional< Value > {
                        if (inputs.size() != 1) {
                            return std::nullopt;
                        }

                        return builder
                            .create< mlir::UnrealizedConversionCastOp >(loc, resultType, inputs)
                            .getResult(0);
                    }
                );
            }

            using mixin_base = tc::mixins< value_category_type_converter >;
            using mixin_base::convert_type_to_type;
            using mixin_base::convert_type_to_types;
        };

#define VAST_DEFINE_REWRITE \
    using base::base; \
\
    logical_result matchAndRewrite( \
        op_t op, typename op_t::Adaptor ops, conversion_rewriter &rewriter \
    ) const override

        template< typename op_t >
        using root_pattern = mlir::OpConversionPattern< op_t >;

        template< typename op_t >
        struct base_pattern : root_pattern< op_t >
        {
            using base = root_pattern< op_t >;

            value_category_type_converter &tc;

            base_pattern(mcontext_t &mctx, value_category_type_converter &tc)
                : base(&mctx), tc(tc) {}

            auto getTypeConverter() const { return &this->tc; }

            static mlir_type element_type(mlir_type t) {
                if (auto as_ptr = mlir::dyn_cast< hl::PointerType >(t)) {
                    return as_ptr.getElementType();
                }
                if (auto as_lvalue = mlir::dyn_cast< hl::LValueType >(t)) {
                    return as_lvalue.getElementType();
                }
                return {};
            }

            static mlir_value iN(auto &bld, auto loc, auto type, auto val) {
                return bld.template create< hl::ConstantOp >(
                    loc, type, llvm::APSInt(llvm::APInt(64, val, true))
                );
            }

            mlir_type convert(mlir_type type) const {
                auto trg = tc.convert_type_to_type(type);
                VAST_ASSERT(trg);
                return *trg;
            }
        };

        template< typename op_t >
        struct memory_allocation : base_pattern< op_t >
        {
            using base = base_pattern< op_t >;

            VAST_DEFINE_REWRITE {
                rewriter.replaceOp(op, allocate(op, rewriter));
                return mlir::success();
            }

            auto allocate(op_t op, auto &rewriter) const {
                auto type = this->tc.convert_type_to_type(op.getType());
                return rewriter.template create< ll::Alloca >(op.getLoc(), *type);
            }
        };

        template< typename op_t >
        struct fn
            : base_pattern< op_t >
            , tc::do_type_conversion_on_op< fn< op_t >, value_category_type_converter >
        {
            using base = base_pattern< op_t >;

            VAST_DEFINE_REWRITE {
                auto func_op = mlir::dyn_cast< mlir::FunctionOpInterface >(op.getOperation());
                if (!func_op) {
                    return mlir::failure();
                }
                return this->replace(func_op, rewriter);
            }
        };

        template< typename op_t >
        struct as_load : base_pattern< op_t >
        {
            using base = base_pattern< op_t >;

            VAST_DEFINE_REWRITE {
                auto ptr = ops.getOperands()[0];

                auto et = this->element_type(ptr.getType());
                VAST_CHECK(et, "{0} was not a pointer!", ptr);
                auto load = rewriter.create< ll::Load >(op.getLoc(), et, ptr);

                rewriter.replaceOp(op, load);

                return mlir::success();
            }
        };

        struct lvalue_to_rvalue_cast : as_load< hl::ImplicitCastOp >
        {
            using op_t = hl::ImplicitCastOp;
            using base = as_load< op_t >;

            VAST_DEFINE_REWRITE {
                if (op.getKind() != hl::CastKind::LValueToRValue) {
                    return mlir::failure();
                }
                return this->base::matchAndRewrite(op, ops, rewriter);
            }
        };

        template< typename op_t >
        struct ignore : base_pattern< op_t >
        {
            using base = base_pattern< op_t >;

            VAST_DEFINE_REWRITE {
                rewriter.replaceOp(op, ops.getOperands()[0]);
                return mlir::success();
            }
        };

        struct array_to_pointer_decay_cast : base_pattern< hl::ImplicitCastOp >
        {
            using op_t = hl::ImplicitCastOp;
            using base = base_pattern< op_t >;

            VAST_DEFINE_REWRITE {
                if (op.getKind() != hl::CastKind::ArrayToPointerDecay) {
                    return mlir::failure();
                }

                auto trg_type = this->convert(op.getResult().getType());
                auto cast     = rewriter.create< hl::ImplicitCastOp >(
                    op.getLoc(), trg_type, ops.getValue(), ops.getKind()
                );
                rewriter.replaceOp(op, cast);

                return mlir::success();
            }
        };

        template< typename op_t >
        struct identity : base_pattern< op_t >
        {
            using base = base_pattern< op_t >;

            VAST_DEFINE_REWRITE {
                auto new_op = rewriter.clone(*op);
                new_op->setOperands(ops.getOperands());
                for (auto v : new_op->getResults()) {
                    v.setType(this->convert(v.getType()));
                }
                rewriter.replaceOp(op, new_op);
                return mlir::success();
            }
        };

        struct subscript : base_pattern< hl::SubscriptOp >
        {
            using op_t = hl::SubscriptOp;

            using base = base_pattern< op_t >;

            VAST_DEFINE_REWRITE {
                auto trg_type = this->convert(op.getResult().getType());
                auto new_op   = rewriter.create< ll::Subscript >(
                    op.getLoc(), trg_type, ops.getArray(), ops.getIndex()
                );
                rewriter.replaceOp(op, new_op);
                return mlir::success();
            }
        };

        template< typename op_t >
        struct store_and_forward_ptr : base_pattern< op_t >
        {
            using base = base_pattern< op_t >;

            VAST_DEFINE_REWRITE {
                rewriter.template create< ll::Store >(
                    op.getLoc(), ops.getElements()[0], ops.getVar()
                );

                rewriter.replaceOp(op, ops.getVar());
                return mlir::success();
            }
        };

        struct prefix_tag
        {};

        struct postfix_tag
        {};

        template< typename Tag >
        constexpr static bool prefix_yield() {
            return std::is_same_v< Tag, prefix_tag >;
        }

        template< typename Tag >
        constexpr static bool postfix_yield() {
            return std::is_same_v< Tag, postfix_tag >;
        }

        template< typename op_t, typename Trg, typename YieldAt >
        struct unary_in_place : base_pattern< op_t >
        {
            using base = base_pattern< op_t >;

            VAST_DEFINE_REWRITE {
                auto arg  = ops.getArg();
                auto type = this->element_type(arg.getType());
                if (!type) {
                    return mlir::failure();
                }

                auto value  = rewriter.create< ll::Load >(op.getLoc(), type, arg);
                auto one    = this->iN(rewriter, op.getLoc(), value.getType(), 1);
                auto adjust = rewriter.create< Trg >(op.getLoc(), type, value, one);

                rewriter.create< ll::Store >(op.getLoc(), adjust, arg);

                auto yielded = [&]() {
                    if constexpr (prefix_yield< YieldAt >()) {
                        return adjust;
                    } else if constexpr (postfix_yield< YieldAt >()) {
                        return value;
                    }
                }();

                rewriter.replaceOp(op, yielded);
                return logical_result::success();
            }

            static void legalize(mlir::ConversionTarget &trg) { trg.addIllegalOp< op_t >(); }
        };

        using unary_in_place_conversions = util::type_list<
            unary_in_place< hl::PreIncOp, hl::AddIOp, prefix_tag >,
            unary_in_place< hl::PostIncOp, hl::AddIOp, postfix_tag >,

            unary_in_place< hl::PreDecOp, hl::SubIOp, prefix_tag >,
            unary_in_place< hl::PostDecOp, hl::SubIOp, postfix_tag > >;

        template< typename op_t, typename Trg >
        struct assign_pattern : base_pattern< op_t >
        {
            using base = base_pattern< op_t >;

            VAST_DEFINE_REWRITE {
                auto lhs = ops.getDst();
                auto rhs = ops.getSrc();

                // TODO(lukas): This should not happen?
                if (rhs.getType().template isa< hl::LValueType >()) {
                    return logical_result::failure();
                }

                auto trg_type = this->convert(op.getSrc().getType());

                // Probably the easiest way to compose this (some template specialization would
                // require a lot of boilerplate).
                auto new_op = [&]() {
                    if constexpr (!std::is_same_v< Trg, void >) {
                        auto load_lhs = rewriter.create< ll::Load >(op.getLoc(), trg_type, lhs);
                        return rewriter.create< Trg >(op.getLoc(), trg_type, load_lhs, rhs);
                    } else {
                        return rhs;
                    }
                }();
                rewriter.create< ll::Store >(op.getLoc(), new_op, lhs);

                // `hl.assign` returns value for cases like `int x = y = 5;`
                rewriter.replaceOp(op, new_op);
                return logical_result::success();
            }

            static void legalize(mlir::ConversionTarget &trg) { trg.addIllegalOp< op_t >(); }
        };

        using assign_conversions = util::type_list<
            assign_pattern< hl::AddIAssignOp, hl::AddIOp >,
            assign_pattern< hl::SubIAssignOp, hl::SubIOp >,
            assign_pattern< hl::MulIAssignOp, hl::MulIOp >,

            assign_pattern< hl::AddFAssignOp, hl::AddFOp >,
            assign_pattern< hl::SubFAssignOp, hl::SubFOp >,
            assign_pattern< hl::MulFAssignOp, hl::MulFOp >,

            assign_pattern< hl::DivSAssignOp, hl::DivSOp >,
            assign_pattern< hl::DivUAssignOp, hl::DivUOp >,
            assign_pattern< hl::DivFAssignOp, hl::DivFOp >,

            assign_pattern< hl::RemSAssignOp, hl::RemSOp >,
            assign_pattern< hl::RemUAssignOp, hl::RemUOp >,
            assign_pattern< hl::RemFAssignOp, hl::RemFOp >,

            assign_pattern< hl::BinOrAssignOp, hl::BinOrOp >,
            assign_pattern< hl::BinAndAssignOp, hl::BinAndOp >,
            assign_pattern< hl::BinXorAssignOp, hl::BinXorOp >,

            assign_pattern< hl::BinShlAssignOp, hl::BinShlOp >,

            assign_pattern< hl::BinLShrAssignOp, hl::BinLShrOp >,
            assign_pattern< hl::BinAShrAssignOp, hl::BinAShrOp >,

            assign_pattern< hl::AssignOp, void > >;

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

        struct fallback : tc::generic_type_converting_pattern< value_category_type_converter >
        {
            using base = tc::generic_type_converting_pattern< value_category_type_converter >;
            using base::base;

            logical_result matchAndRewrite(
                operation op, mlir::ArrayRef< mlir_value > ops, conversion_rewriter &rewriter
            ) const override {
                if (mlir::failed(this->base::matchAndRewrite(op, ops, rewriter))) {
                    return mlir::failure();
                }

                auto convert = [&](auto t) {
                    auto mt = this->get_type_converter().convert_type_to_type(t);
                    VAST_ASSERT(mt);
                    return *mt;
                };

                auto new_op = rewriter.clone(*op);
                new_op->setOperands(ops);
                for (auto v : new_op->getResults()) {
                    v.setType(convert(v.getType()));
                }

                rewriter.replaceOp(op, new_op);
                return mlir::success();
            }
        };

#undef VAST_DEFINE_REWRITE

    } // namespace

    struct type_rewriter : pattern_rewriter
    {
        type_rewriter(mcontext_t *mctx) : pattern_rewriter(mctx) {}
    };

    struct LowerValueCategoriesPass : LowerValueCategoriesBase< LowerValueCategoriesPass >
    {
        using base = LowerValueCategoriesBase< LowerValueCategoriesPass >;

        template< typename... Args >
        void populate(
            util::type_list< Args... >, mlir::RewritePatternSet &patterns,
            mlir::ConversionTarget &trg, mcontext_t &mctx, value_category_type_converter &tc
        ) {
            patterns.add< Args... >(mctx, tc);
            (Args::legalize(trg), ...);
        }

        void runOnOperation() override {
            auto root  = getOperation();
            auto &mctx = getContext();

            value_category_type_converter tc(mctx);

            mlir::RewritePatternSet patterns(&mctx);
            auto trg = mlir::ConversionTarget(mctx);

            patterns.add< fallback >(tc, mctx);
            patterns.add< store_and_forward_ptr< ll::CellInit > >(mctx, tc);
            patterns
                .add< ignore< hl::DeclRefOp >, ignore< hl::Deref >, ignore< hl::AddressOf > >(
                    mctx, tc
                );

            patterns.add< memory_allocation< ll::Cell > >(mctx, tc);
            patterns.add< subscript >(mctx, tc);

            // implicit casts
            patterns.add< lvalue_to_rvalue_cast >(mctx, tc);
            patterns.add< array_to_pointer_decay_cast >(mctx, tc);

            // TODO: This should probably be left to a separate pass.
            // `hl.expr` will get inlined.
            patterns.add< propagate_yield< hl::ExprOp, hl::ValueYieldOp > >(mctx, tc);

            populate(unary_in_place_conversions{}, patterns, trg, mctx, tc);
            populate(assign_conversions{}, patterns, trg, mctx, tc);

            auto is_legal = [&](auto op) {
                return tc.template get_has_legal_return_type< operation >()(op)
                    && tc.template get_has_legal_operand_types< operation >()(op);
            };
            trg.markUnknownOpDynamicallyLegal(is_legal);

            // As we go and replace operands, sometimes it can happen that this cast
            // already will be in form `hl.ptr< T > -> T` instead of `hl.lvalue< T > -> T`.
            // I am not sure why this is happening, quite possibly some pattern is doing
            // something that breaks some invariant, but for now this fix works.
            trg.addDynamicallyLegalOp< hl::ImplicitCastOp >([&](hl::ImplicitCastOp op) {
                return op.getKind() != hl::CastKind::LValueToRValue && is_legal(op);
            });

            trg.addIllegalOp< hl::PreIncOp, hl::PreDecOp, hl::PostIncOp, hl::PreIncOp >();

            // This will never have correct types but we want to have it legal.
            trg.addLegalOp< mlir::UnrealizedConversionCastOp >();

            convert_function_types(tc);
            if (mlir::failed(mlir::applyPartialConversion(root, trg, std::move(patterns)))) {
                return signalPassFailure();
            }
        }

        void convert_function_types(value_category_type_converter &tc) {
            auto root    = getOperation();
            auto pattern = fn< ll::FuncOp >(getContext(), tc);
            auto walker  = [&](mlir::FunctionOpInterface op) {
                type_rewriter bld(&getContext());
                [[maybe_unused]] auto status = pattern.replace(op, bld);
            };

            root->walk(walker);
        }
    };
} // namespace vast::conv

std::unique_ptr< mlir::Pass > vast::createLowerValueCategoriesPass() {
    return std::make_unique< vast::conv::LowerValueCategoriesPass >();
}
