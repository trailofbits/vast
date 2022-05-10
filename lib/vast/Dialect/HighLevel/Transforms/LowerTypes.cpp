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
        VAST_CHECK(static_cast< bool >(t), "Argument of in `contains_hl_type` is not valid.");
        // We need to manually check `t` itself.
        bool found = isHighLevelType(t);
        auto is_hl = [&](auto t) { found |= isHighLevelType(t); };
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
        for (const auto &attr : op->getAttrs())
        {
            // `getType()` is not reliable in reality since for example for `mlir::TypeAttr`
            // it returns none. Lowering of types in attributes will be always best effort.
            if (isHighLevelType(attr.getValue().getType()))
                return true;
            if (auto type_attr = attr.getValue().dyn_cast< mlir::TypeAttr >();
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
        using types_t       = mlir::SmallVector< mlir::Type >;
        using maybe_type_t  = llvm::Optional< mlir::Type >;
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

            addConversion([&](hl::LValueType t) { return this->convert_lvalue_type(t); });

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
        }

        maybe_types_t convert_type(mlir::Type t)
        {
            types_t out;
            if (mlir::succeeded(convertTypes(t, out)))
                return { std::move( out ) };
            return {};
        }

        auto int_type(unsigned bitwidth)
        {
            return mlir::IntegerType::get(&this->mctx, bitwidth);
        }

        auto convert_type() { return [&](auto t) { return this->convert_type(t); }; }
        auto convert_type_to_type()
        {
            return [&](auto t) { return this->convert_type_to_type(t); };
        }
        auto convert_pointer_element_typee()
        {
            return [&](auto t) -> maybe_type_t {
                if (t.template isa< hl::VoidType >()) {
                    return int_type(8u);
                }
                return this->convert_type_to_type(t);
            };
        }

        auto make_int_type()
        {
            return [&](auto t) { return int_type(dl.getTypeSizeInBits(t)); };
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
                    default: VAST_UNREACHABLE("Cannot lower float bitsize {0}", target_bw);
                }
            };
        }

        auto make_ptr_type()
        {
            return [&](auto t) {
                // NOTE(lukas): `none` cannot be memref element type.
                if (t.template isa< mlir::NoneType >())
                    t = mlir::IntegerType::get(&this->mctx, 8);
                return mlir::UnrankedMemRefType::get(t, 0);
            };
        }

        auto make_lvalue_type()
        {
            return [&](auto t) { return hl::LValueType::get(t.getContext(), t); };
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
            return Maybe(t.getElementType()).and_then(convert_pointer_element_typee())
                                            .unwrap()
                                            .and_then(make_ptr_type())
                                            .take_wrapped< maybe_type_t >();
        }

        maybe_type_t convert_lvalue_type(hl::LValueType t)
        {
            return Maybe(t.getElementType()).and_then(convert_type_to_type())
                                            .unwrap()
                                            .and_then(make_lvalue_type())
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

    void convertFunctionSignature(auto &tc,
                                  mlir::FunctionType fty, bool variadic,
                                  mlir::TypeConverter::SignatureConversion &sigconvert)
    {
        for (auto &arg : llvm::enumerate(fty.getInputs()))
        {
            auto cty = tc.convert_type_to_type(arg.value());
            if (!cty)
                return;
            sigconvert.addInputs(arg.index(), { *cty });
        }
    }


    struct AttributeConverter
    {
        mlir::MLIRContext &mctx;
        TypeConverter &tc;

        // `llvm::` instead of `std::` to be uniform with `TypeConverter`
        using maybe_attr_t = llvm::Optional< mlir::Attribute >;

        maybe_attr_t convertAttr(mlir::Attribute attr) const
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

    struct LowerHLTypePatternBase : mlir::ConversionPattern
    {
        AttributeConverter &_attribute_converter;

        LowerHLTypePatternBase(TypeConverter &tc,
                               AttributeConverter &ac,
                               mlir::MLIRContext *mctx)
            : mlir::ConversionPattern(tc, mlir::Pattern::MatchAnyOpTypeTag{}, 1, mctx),
              _attribute_converter(ac)
        {}

        const auto &getAttrConverter() const { return _attribute_converter; }

        void lower_attrs(mlir::Operation *op) const
        {
            mlir::SmallVector< mlir::NamedAttribute > new_attrs;
            for (const auto &attr : op->getAttrs()) {
                auto name = attr.getName();
                auto value = attr.getValue();
                if (auto lowered = this->getAttrConverter().convertAttr(value))
                    new_attrs.emplace_back(name, *lowered);
                else
                    new_attrs.emplace_back(name, value);
            }
            op->setAttrs(new_attrs);
        }
    };

    // `ConversionPattern` provides methods that can use `TypeConverter`, which
    // other patterns do not.
    struct LowerGenericOpType : LowerHLTypePatternBase
    {
        using Base = LowerHLTypePatternBase;
        using Base::Base;

        mlir::LogicalResult matchAndRewrite(
                mlir::Operation *op, mlir::ArrayRef< mlir::Value > ops,
                mlir::ConversionPatternRewriter &rewriter) const override
        {
            if (mlir::isa< mlir::FuncOp >(op))
                return mlir::failure();

            mlir::SmallVector< mlir::Type > rty;
            auto status = this->getTypeConverter()->convertTypes(op->getResultTypes(), rty);
            // TODO(lukas): How to use `llvm::formatv` with `mlir::Operation *`?
            VAST_CHECK(mlir::succeeded(status), "Was not able to type convert.");

            // We just change type, no need to copy everything
            auto lower_op = [&]() {
                if (!mlir::isa< mlir::FuncOp >(op))
                    for (std::size_t i = 0; i < rty.size(); ++i)
                        op->getResult(i).setType(rty[i]);

                // TODO(lukas): Investigate if moving to separate pattern is better
                //              way to do this.
                // TODO(lukas): Other operations that can have block arguments.
                if (auto fn = mlir::dyn_cast_or_null< mlir::FuncOp >(op))
                {
                    mlir::TypeConverter::SignatureConversion sigconvert(fn.getNumArguments());
                    convertFunctionSignature(getAttrConverter().tc,
                                             fn.getType(), false, sigconvert);
                    if (mlir::failed(rewriter.convertRegionTypes(&fn.getBody(),
                                                                 *getTypeConverter(),
                                                                 &sigconvert)))
                    {
                        VAST_UNREACHABLE("Cannot handle failure to update block types.");
                    }
                }
                // For example return type of function can be encoded in attributes
                lower_attrs(op);
            };
            // It has to be done in one "transaction".
            rewriter.updateRootInPlace(op, lower_op);

            return mlir::success();
        }
    };

    struct LowerFuncOpType : LowerHLTypePatternBase
    {
        using Base = LowerHLTypePatternBase;
        using Base::Base;

        mlir::LogicalResult matchAndRewrite(
                mlir::Operation *op, mlir::ArrayRef< mlir::Value > ops,
                mlir::ConversionPatternRewriter &rewriter) const override
        {
            if (!mlir::isa< mlir::FuncOp >(op))
                return mlir::failure();
            return mlir::failure();
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

        patterns.add< LowerGenericOpType,
                      LowerFuncOpType     >(type_converter, attr_converter,
                                            patterns.getContext());

        if (mlir::failed(mlir::applyPartialConversion(
                         op, trg, std::move(patterns))))
            return signalPassFailure();
    }

    mlir::Block &solo_block(mlir::Region &region)
    {
        VAST_ASSERT(region.hasOneBlock());
        return *region.begin();
    }

    // TODO(lukas):
    struct LowerRecordDeclOp : mlir::OpConversionPattern< hl::RecordDeclOp >
    {
        using parent_t = mlir::OpConversionPattern< hl::RecordDeclOp >;

        // TODO(lukas): We most likely no longer need type converter here.
        LowerRecordDeclOp(TypeConverter &tc, mlir::MLIRContext *mctx)
            : parent_t(tc, mctx)
        {}

        std::vector< mlir::Type > collect_field_tys(hl::RecordDeclOp op) const
        {
            std::vector< mlir::Type > out;
            for (auto &maybe_field : solo_block(op.fields()))
            {
                auto field = mlir::dyn_cast< hl::FieldDeclOp >(maybe_field);
                VAST_ASSERT(field);
                out.push_back(field.type());
            }
            return out;
        }

        // TODO(lukas): This is definitely **not** how it should be done.
        //              Rework once links via symbols have api.
        std::vector< hl::TypeDeclOp > fetch_decls(hl::RecordDeclOp op) const
        {
            std::vector< hl::TypeDeclOp > out;
            auto module_op = op->getParentOfType< mlir::ModuleOp >();
            for (auto &x : solo_block(module_op.body()))
            {
                if (auto type_decl = mlir::dyn_cast< hl::TypeDeclOp >(x);
                    type_decl && type_decl.name() == op.name())
                {
                    out.push_back(type_decl);
                }
            }
            return out;
        }

        mlir::LogicalResult matchAndRewrite(
                hl::RecordDeclOp op, hl::RecordDeclOp::Adaptor ops,
                mlir::ConversionPatternRewriter &rewriter) const override
        {
            auto field_tys = collect_field_tys(op);
            auto trg_ty = mlir::TupleType::get(this->getContext(), field_tys);

            rewriter.create< hl::TypeDefOp >(
                    op.getLoc(), op.name(), trg_ty);

            auto type_decls = fetch_decls(op);
            for (auto x : type_decls)
                rewriter.eraseOp(x);

            rewriter.eraseOp(op);
            return mlir::success();
        }
    };

    struct StructsToTuplesPass : StructsToTuplesBase< StructsToTuplesPass >
    {
        void runOnOperation() override
        {
            auto op = this->getOperation();
            auto &mctx = this->getContext();

            // TODO(lukas): Simply inherit and overload to accept everything but that one op.
            // TODO(lukas): Will probably need to emit extracts as well.
            mlir::ConversionTarget trg(mctx);
            trg.addIllegalOp< hl::RecordDeclOp >();
            trg.addLegalOp< hl::TypeDefOp >();

            mlir::RewritePatternSet patterns(&mctx);
            const auto &dl_analysis = this->getAnalysis< mlir::DataLayoutAnalysis >();
            TypeConverter type_converter(dl_analysis.getAtOrAbove(op), mctx);

            patterns.add< LowerRecordDeclOp >(type_converter, patterns.getContext());
            if (mlir::failed(mlir::applyPartialConversion(
                             op, trg, std::move(patterns))))
            {
                return signalPassFailure();
            }
        }
    };
}

std::unique_ptr< mlir::Pass > vast::hl::createLowerHighLevelTypesPass()
{
    return std::make_unique< LowerHighLevelTypesPass >();
}

std::unique_ptr< mlir::Pass > vast::hl::createStructsToTuplesPass()
{
    return std::make_unique< StructsToTuplesPass >();
}
