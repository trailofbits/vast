// Copyright (c) 2021-present, Trail of Bits, Inc.

#include "vast/Dialect/HighLevel/Passes.hpp"

VAST_RELAX_WARNINGS
#include <mlir/Analysis/DataLayoutAnalysis.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
VAST_UNRELAX_WARNINGS

#include "PassesDetails.hpp"

#include "vast/Dialect/HighLevel/HighLevelTypes.hpp"
#include "vast/Dialect/HighLevel/HighLevelOps.hpp"
#include "vast/Dialect/HighLevel/HighLevelAttributes.hpp"

#include "vast/Util/Maybe.hpp"

#include <iostream>

namespace vast::hl
{
    auto get_value() { return [](auto attr) { return attr.getValue(); }; }

    template< typename T >
    auto dyn_cast() { return [](auto x) { return x.template dyn_cast< T >(); }; }

    bool contains_hl_type(mlir::Type t)
    {
        CHECK(static_cast< bool >(t), "Argument of in `contains_hl_type` is not valid.");
        // We need to manually check `t` itself.
        bool found = isHighLevelType(t);
        auto is_hl = [&](auto t)
        {
            if (t.template isa< hl::RecordType >())
                return;
            found |= isHighLevelType(t);
        };
        // If `t` is aggregate, walk over all nested types.
        if (auto is_aggregate = t.dyn_cast< mlir::SubElementTypeInterface >())
            is_aggregate.walkSubTypes(is_hl);
        return found;
    }

    bool contain_hl_type(mlir::TypeRange type_range)
    {
        for (auto x : type_range)
            if (contains_hl_type(x))
                return true;
        return false;
    }

    bool isHighLevelType(mlir::TypeAttr type_attr)
    {
        return Maybe(type_attr).and_then(get_value())
                               .and_then(dyn_cast< mlir::Type >())
                               .keep_if(isHighLevelType)
                               .has_value();
    }

    bool has_hl_typeattr(mlir::Operation *op)
    {
        for (const auto &[_, attr] : op->getAttrs())
        {
            // `getType()` is not reliable in reality since for example for `mlir::TypeAttr`
            // it returns none. Lowering of types in attributes will be always best effort.
            if (isHighLevelType(attr.getType()))
                return true;
            if (auto type_attr = attr.dyn_cast< mlir::TypeAttr >();
                type_attr && contains_hl_type(type_attr.getValue()))
            {
                return true;
            }

        }
        return false;
    }

    bool has_hl_type(mlir::Operation *op)
    {
        return contain_hl_type(op->getResultTypes()) ||
               contain_hl_type(op->getOperandTypes()) ||
               has_hl_typeattr(op);
    }

    bool should_lower(mlir::Operation *op) { return !has_hl_type(op); }

    struct TypeConverter : mlir::TypeConverter {
        using types_t = mlir::SmallVector< mlir::Type >;
        using maybe_type_t = llvm::Optional< mlir::Type >;
        using maybe_types_t = llvm::Optional< types_t >;

        const mlir::DataLayout &dl;
        mlir::MLIRContext &mctx;

        TypeConverter(const mlir::DataLayout &dl_, mlir::MLIRContext &mctx_)
            : mlir::TypeConverter(), dl(dl_), mctx(mctx_)
        {
            // Fallthrough option - we define it first as it seems the framework
            // goes from the last added conversion.
            addConversion([&](mlir::Type t) -> llvm::Optional< mlir::Type > {
                return Maybe(t).keep_if([](auto t){ return !isHighLevelType(t); })
                               .take_wrapped< maybe_type_t >();
            });
            addConversion([&](mlir::Type t) { return this->try_convert_intlike(t); });
            addConversion([&](mlir::Type t) { return this->try_convert_floatlike(t); });

            // Use provided data layout to get the correct type.
            addConversion([&](hl::PointerType t) { return this->convert_ptr_type(t); });
            addConversion([&](mlir::FunctionType t) { return this->convert_fn_type(t); });
            addConversion([&](hl::ConstantArrayType t) {
                    return this->convert_const_arr_type(t);
            });
            // TODO(lukas): This one is tricky, because ideally `hl.void` is "no value".
            //              But if we lowered it such, than we need to remove the previous
            //              value and everything gets more complicated.
            //              This approach should be fine as long as rest of `mlir` accepts
            //              none type.
            addConversion([&](hl::VoidType t) -> maybe_type_t
            {
                    return { mlir::NoneType::get(&mctx) };
            });
            // TODO(lukas): Support properly.
            addConversion([&](hl::RecordType t) { return t; });

        }

        maybe_types_t convert_type(mlir::Type t)
        {
            types_t out;
            if (mlir::succeeded(convertTypes(t, out)))
                return { std::move( out ) };
            return {};
        }

        auto convert_type() { return [&](auto t) { return this->convert_type(t); }; }
        auto convert_type_to_type()
        {
            return [&](auto t) { return this->convert_type_to_type(t); };
        }

        auto make_int_type()
        {
            return [&](auto t) {
                return mlir::IntegerType::get(&this->mctx, dl.getTypeSizeInBits(t));
            };
        }

        auto make_float_type()
        {
            return [&](auto t) {
                auto target_bw = dl.getTypeSizeInBits(t);
                switch (target_bw)
                {
                    case 16: return mlir::FloatType::getF16(&mctx);
                    case 32: return mlir::FloatType::getF32(&mctx);
                    case 64: return mlir::FloatType::getF64(&mctx);
                    case 80: return mlir::FloatType::getF80(&mctx);
                    case 128: return mlir::FloatType::getF128(&mctx);
                    default: UNREACHABLE("Cannot lower float bitsize {0}", target_bw);
                }
            };
        }

        auto make_ptr_type()
        {
            return [&](auto t) { return mlir::UnrankedMemRefType::get(t, 0); };
        }

        maybe_types_t convert_type_to_types(mlir::Type t, std::size_t count = 1)
        {
            return Maybe(t).and_then(convert_type())
                           .keep_if([&](const auto &ts) { return ts->size() == count; })
                           .take_wrapped< maybe_types_t >();
        }

        maybe_type_t convert_type_to_type(mlir::Type t)
        {
            return Maybe(t).and_then([&](auto t){ return this->convert_type_to_types(t, 1); })
                           .and_then([&](auto ts){ return *ts->begin(); })
                           .take_wrapped< maybe_type_t >();
        }


        maybe_type_t try_convert_intlike(mlir::Type t)
        {
            // For now `bool` behaves the same way as any other integer type.
            return Maybe(t).keep_if([](auto t){ return isIntegerType(t) || isBoolType(t); })
                           .and_then(make_int_type())
                           .take_wrapped< maybe_type_t >();
        }

        maybe_type_t try_convert_floatlike(mlir::Type t)
        {
            return Maybe(t).keep_if(isFloatingType)
                           .and_then(make_float_type())
                           .take_wrapped< maybe_type_t >();
        }

        maybe_type_t convert_ptr_type(hl::PointerType t)
        {
            return Maybe(t.getElementType()).and_then(convert_type_to_type())
                                            .unwrap()
                                            .and_then(make_ptr_type())
                                            .take_wrapped< maybe_type_t >();
        }

        maybe_type_t convert_const_arr_type(hl::ConstantArrayType t)
        {
            auto [dim, nested_ty] = t.dim_and_type();
            std::vector< int64_t > coerced_dim;
            for (auto x : dim)
                coerced_dim.push_back(static_cast< int64_t >(x));

            return Maybe(convert_type_to_type(nested_ty))
                    .and_then([&](auto t) { return mlir::MemRefType::get({coerced_dim}, *t); })
                    .take_wrapped< maybe_type_t >();
        }

        maybe_type_t convert_fn_type(mlir::FunctionType t)
        {
            mlir::SmallVector< mlir::Type > aty;
            mlir::SmallVector< mlir::Type > rty;

            auto a_res = convertTypes(t.getInputs(), aty);
            auto r_res = convertTypes(t.getResults(), rty);

            if (mlir::failed(a_res) || mlir::failed(r_res))
                return llvm::None;
            return mlir::FunctionType::get(&mctx, aty, rty);
        }
    };

    struct AttributeConverter
    {
        mlir::MLIRContext &mctx;
        TypeConverter &tc;

        // `llvm::` instead of `std::` to be uniform with `TypeConverter`
        using maybe_attr_t = llvm::Optional< mlir::Attribute >;

        maybe_attr_t convertAttr(mlir::Identifier id, mlir::Attribute attr) const
        {
            if (auto type_attr = attr.dyn_cast< mlir::TypeAttr >())
            {
                return Maybe(type_attr.getValue())
                    .and_then(tc.convert_type_to_type())
                    .unwrap()
                    .and_then(mlir::TypeAttr::get)
                    .take_wrapped< maybe_attr_t >();
            }
            return {};
        }
    };

    // `ConversionPattern` provides methods that can use `TypeConverter`, which
    // other patterns do not.
    struct LowerHLTypePattern : mlir::ConversionPattern
    {
        AttributeConverter &_attribute_converter;

        LowerHLTypePattern(TypeConverter &tc, AttributeConverter &ac, mlir::MLIRContext *mctx)
            : mlir::ConversionPattern(tc, mlir::Pattern::MatchAnyOpTypeTag{}, 1, mctx),
              _attribute_converter(ac)
        {}

        const auto &getAttrConverter() const { return _attribute_converter; }

        // `ops` are remapped operands.
        // `op` is current operation (with old operands).
        // `rewriter` is created by mlir when we start conversion.
        mlir::LogicalResult matchAndRewrite(
                mlir::Operation *op, mlir::ArrayRef< mlir::Value > ops,
                mlir::ConversionPatternRewriter &rewriter) const override
        {
            mlir::SmallVector< mlir::Type > rty;
            auto status = this->getTypeConverter()->convertTypes(op->getResultTypes(), rty);
            // TODO(lukas): How to use `llvm::formatv` with `mlir::Operation *`?
            CHECK(mlir::succeeded(status), "Was not able to type convert.");

            // We just change type, no need to copy everything
            auto lower_op = [&]() {
                for (std::size_t i = 0; i < rty.size(); ++i)
                    op->getResult(i).setType(rty[i]);

                // TODO(lukas): Investigate if moving to separate pattern is better
                //              way to do this.
                // TODO(lukas): Other operations that can have block arguments.
                if (auto fn = mlir::dyn_cast_or_null< mlir::FuncOp >(op))
                    if (mlir::failed(rewriter.convertRegionTypes(&fn.getBody(),
                                                                 *getTypeConverter())))
                        UNREACHABLE("Cannot handle failure to update block types.");
                // For example return type of function can be encoded in attributes
                lower_attrs(op);
            };
            // It has to be done in one "transaction".
            rewriter.updateRootInPlace(op, lower_op);

            return mlir::success();
        }

        void lower_attrs(mlir::Operation *op) const
        {
            mlir::SmallVector< mlir::NamedAttribute > new_attrs;
            for (const auto &[id, attr] : op->getAttrs())
            {
                if (auto lowered = this->getAttrConverter().convertAttr(id, attr))
                    new_attrs.emplace_back(id, *lowered);
                else
                    new_attrs.emplace_back(id, attr);
            }
            op->setAttrs(new_attrs);
        }
    };

    struct LowerHighLevelTypesPass : LowerHighLevelTypesBase< LowerHighLevelTypesPass >
    {
        void runOnOperation() override;
    };

    void LowerHighLevelTypesPass::runOnOperation()
    {
        auto op = this->getOperation();
        auto &mctx = this->getContext();

        mlir::ConversionTarget trg(mctx);
        // We want to check *everything* for presence of hl type
        // that can be lowered.
        trg.markUnknownOpDynamicallyLegal(should_lower);

        mlir::RewritePatternSet patterns(&mctx);
        const auto &dl_analysis = this->getAnalysis< mlir::DataLayoutAnalysis >();
        TypeConverter type_converter(dl_analysis.getAtOrAbove(op), mctx);
        AttributeConverter attr_converter{mctx, type_converter};

        patterns.add< LowerHLTypePattern >(type_converter, attr_converter,
                                           patterns.getContext());

        if (mlir::failed(mlir::applyPartialConversion(
                        op, trg, std::move(patterns))))
            return signalPassFailure();
    }
}


std::unique_ptr< mlir::Pass > vast::hl::createLowerHighLevelTypesPass()
{
  return std::make_unique< LowerHighLevelTypesPass >();
}
