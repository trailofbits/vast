// Copyright (c) 2024-present, Trail of Bits, Inc.

#include "vast/Conversion/Passes.hpp"

VAST_RELAX_WARNINGS
#include <mlir/IR/PatternMatch.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <mlir/IR/BuiltinDialect.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/Rewrite/FrozenRewritePatternSet.h>
VAST_UNRELAX_WARNINGS

#include "../PassesDetails.hpp"

#include "vast/Dialect/HighLevel/HighLevelTypes.hpp"

#include "vast/Dialect/LowLevel/LowLevelOps.hpp"

#include "vast/Util/Common.hpp"

#include "vast/Conversion/Common/Passes.hpp"
#include "vast/Conversion/TypeConverters/TypeConvertingPattern.hpp"

namespace vast::conv
{
    namespace
    {
        struct value_category_type_converter
            : tc::base_type_converter,
              tc::mixins< value_category_type_converter >,
              tc::ConvertFunctionType< value_category_type_converter >
        {
            mlir::MLIRContext &mctx;

            // Do we need data layout? Probably not as everything should be explicit
            // type size wise.
            value_category_type_converter(mcontext_t &mctx)
                : mctx(mctx)
            {
                init();

                addConversion(this->template convert_fn_type< core::FunctionType >());
            }

            using mixin_base = tc::mixins< value_category_type_converter >;
            using mixin_base::convert_type_to_type;
            using mixin_base::convert_type_to_types;

            template< typename T, typename... Args >
            auto make_aggregate_type(Args... args) {
                return [=](mlir_type elem) { return T::get(elem.getContext(), elem, args...); };
            }

            auto convert_lvalue_type() {
                return [&](hl::LValueType type) {
                    return Maybe(type.getElementType())
                        .and_then(convert_type_to_type())
                        .unwrap()
                        .and_then(make_aggregate_type< hl::PointerType >())
                        .template take_wrapped< maybe_type_t >();
                };
            }

            template< typename T >
            auto pointerlike_as_memref()
            {
                return [&](T type) {
                    // It should never happen that we have nested lvalues?
                    auto element_type = this->convert_type_to_type(type.getElementType());
                    return hl::PointerType::get(&mctx, *element_type);
                };
            }

            void init()
            {
                addConversion([&](mlir_type t) { return t; });
                addConversion(pointerlike_as_memref< hl::LValueType >());
                addTargetMaterialization([&](mlir::OpBuilder &builder, mlir::Type resultType,
                                             mlir::ValueRange inputs,
                                             mlir::Location loc) -> std::optional<Value>
                {
                    if (inputs.size() != 1)
                      return std::nullopt;

                    return builder.create<mlir::UnrealizedConversionCastOp>(
                        loc, resultType, inputs).getResult(0);
                });
                addSourceMaterialization([&](mlir::OpBuilder &builder, mlir::Type resultType,
                                             mlir::ValueRange inputs,
                                             mlir::Location loc) -> std::optional<Value>
                {
                    if (inputs.size() != 1)
                      return std::nullopt;

                    return builder.create<mlir::UnrealizedConversionCastOp>(
                        loc, resultType, inputs).getResult(0);
                });
            }
        };

#define REWRITE \
        using base::base; \
\
        logical_result matchAndRewrite(op_t op, typename op_t::Adaptor ops, \
                                       conversion_rewriter &rewriter) const override

        template< typename op_t >
        using root_pattern = mlir::OpConversionPattern< op_t >;

        template< typename op_t >
        struct base_pattern : root_pattern< op_t >
        {
            using base = root_pattern< op_t >;

            value_category_type_converter &tc;

            base_pattern(mcontext_t &mctx, value_category_type_converter &tc)
                : base(&mctx), tc(tc)
            {}

            auto getTypeConverter() const { return &this->tc; }

            static mlir_type element_type(mlir_type t) {
                if (auto as_ptr = mlir::dyn_cast< hl::PointerType >(t))
                    return as_ptr.getElementType();
                if (auto as_lvalue = mlir::dyn_cast< hl::LValueType >(t))
                    return as_lvalue.getElementType();
                return {};
            }

            mlir_type convert(mlir_type type) const
            {
                auto trg = tc.convert_type_to_type(type);
                VAST_ASSERT(trg);
                return *trg;
            }
        };


        template< typename op_t >
        struct memory_allocation : base_pattern< op_t >
        {
            using base = base_pattern< op_t >;

            REWRITE {
                rewriter.replaceOp(op, allocate(op, rewriter));
                return mlir::success();
            }

            auto allocate(op_t op, auto &rewriter) const
            {
                auto type = this->tc.convert_type_to_type(op.getType());
                //auto memref_type = mlir::dyn_cast< mlir::MemRefType >(*type);
                return rewriter.template create< ll::Alloca >(
                    op.getLoc(), *type);
            }

            auto initialize(auto allocation, mlir_value value,
                            auto &rewriter) const
            {
                return rewriter.template create< ll::Store >(
                    allocation.getLoc(), value, allocation);
            }
        };

        template< typename op_t >
        struct allocate_and_propagate : memory_allocation< op_t >
        {
            using base = memory_allocation< op_t >;

            REWRITE {
                auto ptr = this->allocate(op, rewriter);
                rewriter.replaceOp(op, ptr);
                return mlir::success();
            }
        };

        template< typename op_t >
        struct fn : base_pattern< op_t >,
                    tc::do_type_conversion_on_op< fn< op_t >, value_category_type_converter >
        {
            using base = base_pattern< op_t >;

            REWRITE {
                auto func_op = mlir::dyn_cast< mlir::FunctionOpInterface >(op.getOperation());
                if (!func_op)
                    return mlir::failure();
                return this->replace(func_op, rewriter);
            }
        };

        template< typename op_t >
        struct with_store : memory_allocation< op_t >
        {
            using base = memory_allocation< op_t >;

            REWRITE {
                auto allocation = this->allocate(op, rewriter);
                this->initialize(allocation, ops.getOperands()[0], rewriter);
                rewriter.replaceOp(op, allocation);

                return mlir::success();
            }
        };

        template< typename op_t >
        struct as_load : base_pattern< op_t >
        {
            using base = base_pattern< op_t >;

            REWRITE {
                auto ptr = ops.getOperands()[0];

                auto et = this->element_type(ptr.getType());
                VAST_CHECK(et, "{0} was not a pointer!", ptr);
                auto load = rewriter.create< ll::Load >(op.getLoc(), et, ptr);

                rewriter.replaceOp(op, load);

                return mlir::success();
            }
        };

        template< typename op_t >
        struct ignore : base_pattern< op_t >
        {
            using base = base_pattern< op_t >;

            REWRITE {
                rewriter.replaceOp(op, ops.getOperands());
                return mlir::success();
            }
        };

        template< typename op_t >
        struct assign : base_pattern< op_t >
        {
            using base = base_pattern< op_t >;

            REWRITE {
                auto a = ops.getOperands()[0];
                auto b = ops.getOperands()[1];
                rewriter.template create< ll::Store >(
                    op.getLoc(), a, b);
                rewriter.replaceOp(op, a);
                return mlir::success();
            }
        };

        template< typename op_t >
        struct identity : base_pattern< op_t >
        {

            using base = base_pattern< op_t >;

            REWRITE {
                auto new_op = rewriter.clone(*op);
                for (auto v : new_op->getResults())
                    v.setType(this->convert(v.getType()));
                rewriter.replaceOp(op, new_op);
                return mlir::success();
            }
        };


        struct fallback : tc::generic_type_converting_pattern< value_category_type_converter >
        {
            using base = tc::generic_type_converting_pattern< value_category_type_converter >;
            using base::base;

            logical_result matchAndRewrite(operation op, mlir::ArrayRef< mlir_value > ops,
                                           conversion_rewriter &rewriter) const override
            {
                if (mlir::failed(this->base::matchAndRewrite(op, ops, rewriter)))
                    return mlir::failure();

                auto update = [&] {
                    op->setOperands(ops);
                };
                rewriter.updateRootInPlace(op, update);
                return mlir::success();

            }
        };
    } // namespace

    struct type_rewriter : pattern_rewriter {
        type_rewriter(mcontext_t *mctx) : pattern_rewriter(mctx) {}
    };

    struct LowerValueCategoriesPass : LowerValueCategoriesBase< LowerValueCategoriesPass >
    {
        using base = LowerValueCategoriesBase< LowerValueCategoriesPass >;

        void runOnOperation() override
        {
            auto root = getOperation();
            auto &mctx = getContext();

            value_category_type_converter tc(mctx);

            mlir::RewritePatternSet patterns(&mctx);

            patterns.add< fallback >(tc, mctx);
            patterns.add< with_store< ll::ArgAlloca > >(mctx, tc);
            patterns.add< ignore< hl::DeclRefOp > >(mctx, tc);
            patterns.add< as_load< hl::ImplicitCastOp > >(mctx, tc);
            patterns.add< ignore< hl::Deref > >(mctx, tc);

            patterns.add< assign< hl::AssignOp > >(mctx, tc);
            patterns.add< identity< ll::UninitializedVar > >(mctx, tc);


            auto trg = mlir::ConversionTarget(mctx);

            auto is_legal = [&](auto op)
            {
                return tc.template get_has_legal_return_type< operation >()(op) &&
                       tc.template get_has_legal_operand_types< operation >()(op);
            };
            trg.markUnknownOpDynamicallyLegal(is_legal);

            // As we go and replace operands, sometimes it can happen that this cast
            // already will be in form `hl.ptr< T > -> T` instead of `hl.lvalue< T > -> T`.
            // I am not sure why this is happening, quite possibly some pattern is doing
            // something that breaks some invariant, but for now this fix works.
            trg.addDynamicallyLegalOp< hl::ImplicitCastOp >([&](hl::ImplicitCastOp op) {
                return op.getKind() != hl::CastKind::LValueToRValue;
            });

            convert_function_types(tc);
            if (mlir::failed(mlir::applyPartialConversion(root, trg, std::move(patterns))))
                return signalPassFailure();
        }

        void convert_function_types(value_category_type_converter &tc)
        {
            auto root = getOperation();
            auto pattern = fn< ll::FuncOp >(getContext(), tc);
            auto walker = [&](mlir::FunctionOpInterface op)
            {
                type_rewriter bld(&getContext());
                [[maybe_unused]] auto status = pattern.replace(op, bld);
            };

            root->walk(walker);
        }
    };
} // namespace vast::conv


std::unique_ptr< mlir::Pass > vast::createLowerValueCategoriesPass()
{
    return std::make_unique< vast::conv::LowerValueCategoriesPass >();
}
