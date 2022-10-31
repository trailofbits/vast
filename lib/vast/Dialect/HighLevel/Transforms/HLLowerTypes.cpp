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

#include "vast/Dialect/HighLevel/HighLevelAttributes.hpp"
#include "vast/Dialect/HighLevel/HighLevelTypes.hpp"
#include "vast/Dialect/HighLevel/HighLevelOps.hpp"

#include "vast/Util/Maybe.hpp"
#include "vast/Util/TypeConverter.hpp"

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
            addConversion([&](hl::ArrayType t) {
                    return this->convert_arr_type(t);
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

        // TODO(lukas): Take optional to denote that is may be `Signless`.
        auto int_type(unsigned bitwidth, bool is_signed)
        {
            auto signedness = [=]()
            {
                if (is_signed)
                    return mlir::IntegerType::SignednessSemantics::Signed;
                return mlir::IntegerType::SignednessSemantics::Unsigned;
            }();
            return mlir::IntegerType::get(&this->mctx, bitwidth, signedness);
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
                    return int_type(8u, mlir::IntegerType::SignednessSemantics::Signless);
                }
                return this->convert_type_to_type(t);
            };
        }

        auto make_int_type(bool is_signed)
        {
            return [=](auto t)
            {
                return int_type(dl.getTypeSizeInBits(t), is_signed);
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
                    default: VAST_UNREACHABLE("Cannot lower float bitsize {0}", target_bw);
                }
            };
        }

        auto make_ptr_type(auto quals)
        {
            return [=](auto t)
            {
                return PointerType::get(t.getContext(), t, quals);
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
            if (!isIntegerType(t) && !isBoolType(t))
                return {};

            return Maybe(t).and_then(make_int_type(isSigned(t)))
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
            return Maybe(t.getElementType())
                .and_then(convert_pointer_element_typee())
                .unwrap()
                .and_then(make_ptr_type(t.getQuals()))
                .take_wrapped< maybe_type_t >();
        }

        maybe_type_t convert_lvalue_type(hl::LValueType t)
        {
            return Maybe(t.getElementType()).and_then(convert_type_to_type())
                                            .unwrap()
                                            .and_then(make_lvalue_type())
                                            .take_wrapped< maybe_type_t >();
        }

        maybe_type_t convert_arr_type(hl::ArrayType arr)
        {
            auto [dims, nested_ty] = arr.dim_and_type();
            std::vector< int64_t > coerced_dim;
            for (auto dim : dims) {
                if (dim.hasValue()) {
                    coerced_dim.push_back(dim.getValue());
                } else {
                    coerced_dim.push_back(-1 /* unknown dim */ );
                }
            }

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

    // Get SignatureConversion if all the sub-conversion are successful, no value otherwise.
    auto get_fn_signature(auto &&tc, FuncOp fn, bool variadic)
        -> std::optional< mlir::TypeConverter::SignatureConversion >
    {
        mlir::TypeConverter::SignatureConversion sigconvert(fn.getNumArguments());
        for (auto arg : llvm::enumerate(fn.getFunctionType().getInputs()))
        {
            mlir::SmallVector< mlir::Type, 2 > converted;
            auto cty = tc.convertType(arg.value(), converted);
            if (mlir::failed(cty))
                return {};
            sigconvert.addInputs(arg.index(), converted);
        }
        return { std::move(sigconvert) };
    }


    struct AttributeConverter
    {
        mlir::MLIRContext &mctx;
        TypeConverter &tc;

        // `llvm::` instead of `std::` to be uniform with `TypeConverter`
        using maybe_attr_t = llvm::Optional< mlir::Attribute >;

        template< typename A, typename ... Args >
        auto make_hl_attr( Args && ... args ) const
        {
            // Expected cheap values are passed around, otherwise perfectly forward.
            return [=](auto type)
            {
                return A::get(type, args ...);
            };
        }

        template< typename Attr, typename ... Rest >
        maybe_attr_t hl_attr_conversion(mlir::Attribute attr) const
        {
            if (auto hl_attr = attr.dyn_cast< Attr >())
            {
                return Maybe(hl_attr.getType())
                    .and_then(tc.convert_type_to_type())
                    .unwrap()
                    .and_then(make_hl_attr< Attr >(hl_attr.getValue()))
                    .template take_wrapped< maybe_attr_t >();
            }
            if constexpr (sizeof ... (Rest) != 0)
                return hl_attr_conversion< Rest ... >(attr);
            return {};
        }

        maybe_attr_t convertAttr(mlir::Attribute attr) const
        {
            if (auto out = hl_attr_conversion< BooleanAttr, IntegerAttr, FloatAttr >(attr))
                return out;

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
        TypeConverter &tc;
        AttributeConverter &_attribute_converter;

        LowerHLTypePatternBase(TypeConverter &tc_,
                               AttributeConverter &ac,
                               mlir::MLIRContext *mctx)
            : mlir::ConversionPattern(tc_, mlir::Pattern::MatchAnyOpTypeTag{}, 1, mctx),
              tc(tc_),
              _attribute_converter(ac)

        {}

        // NOTE(lukas): Is not a virtual function.
        TypeConverter *getTypeConverter() const { return &tc; }
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

        template< typename Filter >
        auto lower_attrs(mlir::ArrayRef< mlir::NamedAttribute > attrs, Filter &&filter) const
        -> mlir::SmallVector< mlir::NamedAttribute, 4 >
        {
            mlir::SmallVector< mlir::NamedAttribute, 4 > out;
            for (const auto &attr : attrs)
            {
                if (filter(attr))
                    // TODO(lukas): Converter should accept & reconstruct NamedAttributes.
                    if (auto x = getAttrConverter().convertAttr(attr.getValue()))
                        out.emplace_back(attr.getName(), *x);
            }
            return out;
        }

        auto lower_attrs(mlir::ArrayRef< mlir::NamedAttribute > attrs) const
        {
            return lower_attrs(attrs, [](const auto &) { return true; });
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
            if (mlir::isa< FuncOp >(op))
                return mlir::failure();

            mlir::SmallVector< mlir::Type > rty;
            auto status = this->getTypeConverter()->convertTypes(op->getResultTypes(), rty);
            // TODO(lukas): How to use `llvm::formatv` with `mlir::Operation *`?
            VAST_CHECK(mlir::succeeded(status), "Was not able to type convert.");

            // We just change type, no need to copy everything
            auto lower_op = [&]()
            {
                for (std::size_t i = 0; i < rty.size(); ++i)
                    op->getResult(i).setType(rty[i]);
                // Return types can be encoded as attrs.
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


        // This is a very random list that is pulled from sources of mlir of version
        // llvm-14.
        static bool is_fn_attr(mlir::NamedAttribute attr)
        {
            auto name = attr.getName();
            if (name == mlir::SymbolTable::getSymbolAttrName() ||
                name == mlir::FunctionOpInterface::getTypeAttrName() ||
                name == "std.varargs")
            {
                return false;
            }

            // Expected the checks will expand.
            return true;
        }

        auto lower_fn_attrs(FuncOp fn) const
        {
            return this->Base::lower_attrs(fn->getAttrs(), is_fn_attr);
        }

        using attrs_t = mlir::SmallVector< mlir::Attribute, 4 >;
        using maybe_attrs_t = std::optional< attrs_t >;
        using signature_conversion_t = mlir::TypeConverter::SignatureConversion;

        attrs_t lower_args_attrs(FuncOp fn, mlir::ArrayAttr arg_dict,
                                 const signature_conversion_t &signature) const
        {
            attrs_t new_attrs;
            for (std::size_t i = 0; i < fn.getNumArguments(); ++i)
            {
                auto mapping = signature.getInputMapping(i);
                for (std::size_t j = 0; j < mapping->size; ++j)
                    new_attrs.push_back(arg_dict[i]);
            }
            return new_attrs;
        }

        maybe_attrs_t lower_args_attrs(FuncOp fn, const signature_conversion_t &signature) const
        {
            if (auto arg_dict = fn.getAllArgAttrs())
                return { lower_args_attrs(fn, arg_dict, signature) };
            return {};
        }

        // As the reference how to lower functions, the `StandardToLLVM` conversion
        // is used.
        // But basically we need to copy the function (maybe just change in-place is possible?)
        // with the converted function type -> copy body -> fix arguments of the entry region.
        mlir::LogicalResult matchAndRewrite(
            mlir::Operation *op, mlir::ArrayRef< mlir::Value > ops,
            mlir::ConversionPatternRewriter &rewriter) const override
        {
            auto fn = mlir::dyn_cast< FuncOp >(op);
            if (!fn)
                return mlir::failure();

            auto sigconvert = get_fn_signature(*getTypeConverter(), fn, false);
            if (!sigconvert)
            {
                return mlir::failure();
            }

            auto attributes = lower_fn_attrs(fn);
            if (auto arg_attrs = lower_args_attrs(fn, *sigconvert))
            {
                auto as_attr = rewriter.getNamedAttr(
                        mlir::FunctionOpInterface::getArgDictAttrName(),
                        rewriter.getArrayAttr(*arg_attrs));
                attributes.push_back(as_attr);
            }

            auto maybe_fn_type = getTypeConverter()->convert_type_to_type(fn.getFunctionType());
            if (!maybe_fn_type)
                return mlir::failure();
            auto fn_type = maybe_fn_type->dyn_cast< mlir::FunctionType >();
            if (!fn_type)
                return mlir::failure();

            // Create new function with converted type
            FuncOp new_fn = rewriter.create< FuncOp >(
                fn.getLoc(), fn.getName(), fn_type, fn.getLinkage(), attributes
            );
            new_fn.setVisibility(mlir::SymbolTable::Visibility::Private);

            // Copy the old region - it will have incorrect arguments (`BlockArgument` on
            // entry `Block`.
            rewriter.inlineRegionBefore(fn.getBody(), new_fn.getBody(), new_fn.end());

            // NOTE(lukas): This may break the contract that all modifications happen
            //              via rewriter.
            util::convert_region_types(fn, new_fn, *sigconvert);

            rewriter.eraseOp(fn);
            return mlir::success();
        }
    };

    struct HLLowerTypesPass : HLLowerTypesBase< HLLowerTypesPass >
    {
        void runOnOperation() override;
    };

    void HLLowerTypesPass::runOnOperation()
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
    struct LowerStructDeclOp : mlir::OpConversionPattern< hl::StructDeclOp >
    {
        using parent_t = mlir::OpConversionPattern< hl::StructDeclOp >;

        // TODO(lukas): We most likely no longer need type converter here.
        LowerStructDeclOp(TypeConverter &tc, mlir::MLIRContext *mctx)
            : parent_t(tc, mctx)
        {}

        std::vector< mlir::Type > collect_field_tys(hl::StructDeclOp op) const
        {
            std::vector< mlir::Type > out;
            for (auto &maybe_field : solo_block(op.getFields()))
            {
                auto field = mlir::dyn_cast< hl::FieldDeclOp >(maybe_field);
                VAST_ASSERT(field);
                out.push_back(field.getType());
            }
            return out;
        }

        // TODO(lukas): This is definitely **not** how it should be done.
        //              Rework once links via symbols have api.
        std::vector< hl::TypeDeclOp > fetch_decls(hl::StructDeclOp op) const
        {
            std::vector< hl::TypeDeclOp > out;
            auto module_op = op->getParentOfType< mlir::ModuleOp >();
            for (auto &x : solo_block(module_op.getBodyRegion()))
            {
                if (auto type_decl = mlir::dyn_cast< hl::TypeDeclOp >(x);
                    type_decl && type_decl.getName() == op.getName())
                {
                    out.push_back(type_decl);
                }
            }
            return out;
        }

        mlir::LogicalResult matchAndRewrite(
                hl::StructDeclOp op, hl::StructDeclOp::Adaptor ops,
                mlir::ConversionPatternRewriter &rewriter) const override
        {
            auto field_tys = collect_field_tys(op);
            auto trg_ty = mlir::TupleType::get(this->getContext(), field_tys);

            rewriter.create< hl::TypeDefOp >(op.getLoc(), op.getName(), trg_ty);

            auto type_decls = fetch_decls(op);
            for (auto x : type_decls)
                rewriter.eraseOp(x);

            rewriter.eraseOp(op);
            return mlir::success();
        }
    };

    struct ConversionTargetBuilder
    {
        using self_t = ConversionTargetBuilder;
        mlir::ConversionTarget trg;

        ConversionTargetBuilder(mlir::MLIRContext &mctx) : trg(mctx) {}

        auto take() { return std::move(trg); }

        auto _illegal()
        {
            return [&]< typename O >() { trg.addIllegalOp< O >(); };
        }

        template< typename O, typename ... Os, typename Fn >
        self_t &_rec(Fn &&fn)
        {
            fn.template operator()< O >();
            if constexpr (sizeof ... (Os) == 0)
                return *this;
            else
                return _rec< Os ... >(std::forward< Fn >(fn));
        }

        template< typename O, typename ... Os >
        self_t &illegal() { return _rec< O, Os ...>(_illegal()); }

        self_t&unkown_as_legal()
        {
            trg.markUnknownOpDynamicallyLegal([](auto){ return true; });
            return *this;
        }
    };

    template< typename Pass >
    struct PassUtils : Pass
    {
        const mlir::DataLayout &get_data_layout()
        {
            const auto &dl_analysis = this->template getAnalysis< mlir::DataLayoutAnalysis >();
            return dl_analysis.getAtOrAbove(this->getOperation());
        }

        TypeConverter make_type_converter()
        {
            return TypeConverter(get_data_layout(), this->getContext());
        }
    };

    struct HLStructsToTuplesPass : HLStructsToTuplesBase< HLStructsToTuplesPass >
    {
        void runOnOperation() override
        {
            auto op = this->getOperation();
            auto &mctx = this->getContext();

            // TODO(lukas): Simply inherit and overload to accept everything but that one op.
            // TODO(lukas): Will probably need to emit extracts as well.
            mlir::ConversionTarget trg(mctx);
            trg.addIllegalOp< hl::StructDeclOp >();
            trg.addLegalOp< hl::TypeDefOp >();

            mlir::RewritePatternSet patterns(&mctx);
            const auto &dl_analysis = this->getAnalysis< mlir::DataLayoutAnalysis >();
            TypeConverter type_converter(dl_analysis.getAtOrAbove(op), mctx);

            patterns.add< LowerStructDeclOp >(type_converter, patterns.getContext());
            if (mlir::failed(mlir::applyPartialConversion(
                             op, trg, std::move(patterns))))
            {
                return signalPassFailure();
            }
        }
    };

    struct HLLowerEnumsPass : PassUtils< HLLowerEnumsBase< HLLowerEnumsPass > >
    {
        void runOnOperation() override
        {
            auto op = this->getOperation();
            auto &mctx = this->getContext();

            mlir::RewritePatternSet patterns(&mctx);

            auto trg = ConversionTargetBuilder(mctx)
                .unkown_as_legal()
                .illegal< hl::EnumDeclOp >()
                .take();

            auto tc = this->make_type_converter();

            if (mlir::failed(mlir::applyPartialConversion(op, trg, std::move(patterns))))
                return signalPassFailure();
        }
    };
}

std::unique_ptr< mlir::Pass > vast::hl::createHLLowerEnumsPass()
{
    return std::make_unique< HLLowerEnumsPass >();
}

std::unique_ptr< mlir::Pass > vast::hl::createHLLowerTypesPass()
{
    return std::make_unique< HLLowerTypesPass >();
}

std::unique_ptr< mlir::Pass > vast::hl::createHLStructsToTuplesPass()
{
    return std::make_unique< HLStructsToTuplesPass >();
}
