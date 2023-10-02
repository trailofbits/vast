// Copyright (c) 2021-present, Trail of Bits, Inc.

#include "vast/Dialect/HighLevel/Passes.hpp"

VAST_RELAX_WARNINGS
#include <mlir/Analysis/DataLayoutAnalysis.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/BuiltinAttributeInterfaces.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
VAST_UNRELAX_WARNINGS

#include "PassesDetails.hpp"

#include "vast/Dialect/HighLevel/HighLevelAttributes.hpp"
#include "vast/Dialect/HighLevel/HighLevelOps.hpp"
#include "vast/Dialect/HighLevel/HighLevelTypes.hpp"

#include "vast/Dialect/Core/CoreTypes.hpp"
#include "vast/Dialect/Core/CoreAttributes.hpp"

#include "vast/Conversion/Common/Types.hpp"

#include "vast/Util/Maybe.hpp"
#include "vast/Util/TypeConverter.hpp"

#include <algorithm>
#include <iostream>

namespace vast::hl {
    auto get_value() {
        return [](auto attr) { return attr.getValue(); };
    }

    template< typename T >
    auto dyn_cast() {
        return [](auto x) { return mlir::dyn_cast< T >(x); };
    }

    auto contains_hl_type = [] (mlir_type t) -> bool {
        VAST_CHECK(static_cast< bool >(t), "Argument of in `contains_hl_type` is not valid.");
        // We need to manually check `t` itself.
        bool found = isHighLevelType(t);
        auto is_hl = [&] (auto t) { found |= isHighLevelType(t); };
        // If `t` is aggregate, walk over all nested types.
        if (auto aggr = mlir::dyn_cast< mlir::SubElementTypeInterface >(t)) {
            aggr.walkSubTypes(is_hl);
        }
        return found;
    };

    bool contain_hl_type(mlir::TypeRange rng) {
        return std::any_of(rng.begin(), rng.end(), contains_hl_type);
    }

    bool contain_hl_type(llvm::ArrayRef<mlir_type> rng) {
        return contain_hl_type(mlir::TypeRange(rng));
    }


    bool isHighLevelType(mlir::TypeAttr type_attr) {
        return Maybe(type_attr)
            .and_then(get_value())
            .and_then(dyn_cast< mlir_type >())
            .keep_if(isHighLevelType)
            .has_value();
    }

    bool has_hl_typeattr(operation op) {
        for (const auto &attr : op->getAttrs()) {
            // `getType()` is not reliable in reality since for example for `mlir::TypeAttr`
            // it returns none. Lowering of types in attributes will be always best effort.
            auto typed_attr = mlir::dyn_cast< mlir::TypedAttr >(attr.getValue());
            if (typed_attr && isHighLevelType(typed_attr.getType())) {
                return true;
            }

            if (auto type_attr = attr.getValue().dyn_cast< mlir::TypeAttr >();
                type_attr && contains_hl_type(type_attr.getValue()))
            {
                return true;
            }
        }
        return false;
    }

    bool has_hl_function_type(operation op) {
        if (auto fn = mlir::dyn_cast< hl::FuncOp >(op)) {
            return contain_hl_type(fn.getArgumentTypes())
                || contain_hl_type(fn.getResultTypes());
        }

        return false;
    }

    bool has_hl_type(operation op) {
        return contain_hl_type(op->getResultTypes())
            || contain_hl_type(op->getOperandTypes())
            || has_hl_function_type(op)
            || has_hl_typeattr(op);
    }

    struct TypeConverter : mlir::TypeConverter
    {
        const mlir::DataLayout &dl;
        mlir::MLIRContext &mctx;

        TypeConverter(const mlir::DataLayout &dl, mcontext_t &mctx)
            : mlir::TypeConverter(), dl(dl), mctx(mctx)
        {
            // Fallthrough option - we define it first as it seems the framework
            // goes from the last added conversion.
            addConversion([&](mlir_type t) -> maybe_type_t {
                return Maybe(t)
                    .keep_if([](auto t) { return !isHighLevelType(t); })
                    .take_wrapped< maybe_type_t >();
            });
            addConversion([&](mlir_type t) { return this->try_convert_intlike(t); });
            addConversion([&](mlir_type t) { return this->try_convert_floatlike(t); });

            addConversion([&](hl::LValueType t) { return this->convert_lvalue_type(t); });
            addConversion([&](hl::DecayedType t) { return this->convert_decayed_type(t); });

            // Use provided data layout to get the correct type.
            addConversion([&](hl::PointerType t) { return this->convert_ptr_type(t); });
            addConversion([&](hl::ArrayType t) { return this->convert_arr_type(t); });
            // TODO(lukas): This one is tricky, because ideally `hl.void` is "no value".
            //              But if we lowered it such, than we need to remove the previous
            //              value and everything gets more complicated.
            //              This approach should be fine as long as rest of `mlir` accepts
            //              none type.
            addConversion([&](hl::VoidType t) -> maybe_type_t {
                return { vast::core::VoidType::get(&mctx) };
            });
        }

        maybe_types_t convert_type(mlir_type t) {
            types_t out;
            if (mlir::succeeded(convertTypes(t, out))) {
                return { std::move(out) };
            }
            return {};
        }

        // TODO(lukas): Take optional to denote that is may be `Signless`.
        auto int_type(unsigned bitwidth, bool is_signed) {
            auto signedness = [=]() {
                if (is_signed)
                    return mlir::IntegerType::SignednessSemantics::Signed;
                return mlir::IntegerType::SignednessSemantics::Unsigned;
            }();
            return mlir::IntegerType::get(&this->mctx, bitwidth, signedness);
        }

        auto convert_type() {
            return [&](auto t) { return this->convert_type(t); };
        }

        auto convert_type_to_type() {
            return [&](auto t) { return this->convert_type_to_type(t); };
        }

        auto convert_pointer_element_typee() {
            return [&](auto t) -> maybe_type_t {
                if (t.template isa< hl::VoidType >()) {
                    return int_type(8u, mlir::IntegerType::SignednessSemantics::Signless);
                }
                return this->convert_type_to_type(t);
            };
        }

        auto make_int_type(bool is_signed) {
            return [=, this](auto t) { return int_type(dl.getTypeSizeInBits(t), is_signed); };
        }

        auto make_float_type() {
            return [&](auto t) {
                auto target_bw = dl.getTypeSizeInBits(t);
                switch (target_bw) {
                    case 16:
                        return mlir::FloatType::getF16(&mctx);
                    case 32:
                        return mlir::FloatType::getF32(&mctx);
                    case 64:
                        return mlir::FloatType::getF64(&mctx);
                    case 80:
                        return mlir::FloatType::getF80(&mctx);
                    case 128:
                        return mlir::FloatType::getF128(&mctx);
                    default:
                        VAST_UNREACHABLE("Cannot lower float bitsize {0}", target_bw);
                }
            };
        }

        auto make_ptr_type(auto quals) {
            return [=](auto t) { return PointerType::get(t.getContext(), t, quals); };
        }

        auto make_lvalue_type() {
            return [&](auto t) { return hl::LValueType::get(t.getContext(), t); };
        }

        maybe_types_t convert_type_to_types(mlir_type t, std::size_t count = 1) {
            return Maybe(t)
                .and_then(convert_type())
                .keep_if([&](const auto &ts) { return ts->size() == count; })
                .take_wrapped< maybe_types_t >();
        }

        maybe_type_t convert_type_to_type(mlir_type t) {
            return Maybe(t)
                .and_then([&](auto t) { return this->convert_type_to_types(t, 1); })
                .and_then([&](auto ts) { return *ts->begin(); })
                .take_wrapped< maybe_type_t >();
        }

        maybe_type_t try_convert_intlike(mlir_type t) {
            // For now `bool` behaves the same way as any other integer type.
            if (!isIntegerType(t) && !isBoolType(t)) {
                return {};
            }

            return Maybe(t).and_then(make_int_type(isSigned(t))).take_wrapped< maybe_type_t >();
        }

        maybe_type_t try_convert_floatlike(mlir_type t) {
            return Maybe(t)
                .keep_if(isFloatingType)
                .and_then(make_float_type())
                .take_wrapped< maybe_type_t >();
        }

        maybe_type_t convert_decayed_type(hl::DecayedType t) {
            return Maybe(t.getElementType())
                .and_then(convert_type_to_type())
                .unwrap()
                .take_wrapped< maybe_type_t >();
        }

        maybe_type_t convert_ptr_type(hl::PointerType t) {
            return Maybe(t.getElementType())
                .and_then(convert_pointer_element_typee())
                .unwrap()
                .and_then(make_ptr_type(t.getQuals()))
                .take_wrapped< maybe_type_t >();
        }

        maybe_type_t convert_lvalue_type(hl::LValueType t) {
            return Maybe(t.getElementType())
                .and_then(convert_type_to_type())
                .unwrap()
                .and_then(make_lvalue_type())
                .take_wrapped< maybe_type_t >();
        }

        maybe_type_t convert_arr_type(hl::ArrayType arr) {
            auto [dims, nested_ty] = arr.dim_and_type();
            std::vector< int64_t > coerced_dim;
            for (auto dim : dims) {
                if (dim.has_value()) {
                    coerced_dim.push_back(dim.value());
                } else {
                    coerced_dim.push_back(mlir::ShapedType::kDynamic);
                }
            }

            return Maybe(convert_type_to_type(nested_ty))
                .and_then([&](auto t) { return mlir::MemRefType::get({ coerced_dim }, *t); })
                .take_wrapped< maybe_type_t >();
        }
    };

    struct LowerHighLevelOpType : mlir::ConversionPattern
    {
        using Base = mlir::ConversionPattern;
        using Base::Base;

        LowerHighLevelOpType(TypeConverter &tc, mcontext_t *mctx)
            : Base(tc, mlir::Pattern::MatchAnyOpTypeTag{}, 1, mctx)
        {}

        template< typename attrs_list >
        maybe_attr_t high_level_typed_attr_conversion(mlir::Attribute attr) const {
            using attr_t = typename attrs_list::head;
            using rest_t = typename attrs_list::tail;

            if (auto typed = mlir::dyn_cast< attr_t >(attr)) {
                if constexpr (std::same_as< attr_t, core::VoidAttr>) {
                    return Maybe(typed.getType())
                        .and_then([&] (auto type) {
                            return getTypeConverter()->convertType(type);
                        })
                        .and_then([&] (auto type) {
                            return core::VoidAttr::get(type.getContext(), type);
                        })
                        .template take_wrapped< maybe_attr_t >();
                } else {
                return Maybe(typed.getType())
                    .and_then([&] (auto type) {
                        return getTypeConverter()->convertType(type);
                    })
                    .and_then([&] (auto type) {
                        return attr_t::get(type, typed.getValue());
                    })
                    .template take_wrapped< maybe_attr_t >();
                }
            }

            if constexpr (attrs_list::size != 1) {
                return high_level_typed_attr_conversion< rest_t >(attr);
            } else {
                return std::nullopt;
            }
        }

        auto convert_high_level_typed_attr() const {
            return [&] (mlir::Attribute attr) {
                return high_level_typed_attr_conversion< core::typed_attrs >(attr);
            };
        }

        logical_result matchAndRewrite(
            operation op, llvm::ArrayRef< mlir_value > ops,
            conversion_rewriter &rewriter
        ) const override {
            if (mlir::isa< FuncOp >(op)) {
                return mlir::failure();
            }

            auto &tc = *getTypeConverter();

            mlir::SmallVector< mlir_type > rty;
            auto status = tc.convertTypes(op->getResultTypes(), rty);
            // TODO(lukas): How to use `llvm::formatv` with `operation `?
            VAST_CHECK(mlir::succeeded(status), "Was not able to type convert.");

            // We just change type, no need to copy everything
            auto lower_op = [&]() {
                for (std::size_t i = 0; i < rty.size(); ++i) {
                    op->getResult(i).setType(rty[i]);
                }

                mlir::AttrTypeReplacer replacer;
                replacer.addReplacement(tc::convert_type_attr(tc));
                replacer.addReplacement(convert_high_level_typed_attr());
                replacer.recursivelyReplaceElementsIn(op, true /* replace attrs */);
            };
            // It has to be done in one "transaction".
            rewriter.updateRootInPlace(op, lower_op);

            return mlir::success();
        }
    };

    struct LowerFuncOpType : mlir::OpConversionPattern< FuncOp >
    {
        using Base = mlir::OpConversionPattern< FuncOp >;
        using Base::Base;

        using Base::getTypeConverter;

        // As the reference how to lower functions, the `StandardToLLVM`
        // conversion is used.
        //
        // But basically we need to copy the function with the converted
        // function type -> copy body -> fix arguments of the entry region.
        logical_result matchAndRewrite(
            FuncOp fn, OpAdaptor adaptor, conversion_rewriter &rewriter
        ) const override {
            auto fty = adaptor.getFunctionType();
            auto &tc = *getTypeConverter();

            tc::signature_conversion_t sigconvert(fty.getNumInputs());
            if (mlir::failed(tc.convertSignatureArgs(fty.getInputs(), sigconvert))) {
                return mlir::failure();
            }

            llvm::SmallVector< mlir_type, 1 > results;
            if (mlir::failed(tc.convertTypes(fty.getResults(), results))) {
                return mlir::failure();
            }

            auto params = sigconvert.getConvertedTypes();

            auto new_type = core::FunctionType::get(
                rewriter.getContext(), params, results, fty.isVarArg()
            );

            // TODO deal with function attribute types

            rewriter.updateRootInPlace(fn, [&] {
                fn.setType(new_type);
                for (auto [ty, param] : llvm::zip(params, fn.getBody().getArguments())) {
                    param.setType(ty);
                }
            });

            return mlir::success();
        }
    };

    struct HLLowerTypesPass : HLLowerTypesBase< HLLowerTypesPass >
    {
        void runOnOperation() override {
            auto op    = this->getOperation();
            auto &mctx = this->getContext();

            mlir::ConversionTarget trg(mctx);
            // We want to check *everything* for presence of hl type
            // that can be lowered.
            trg.markUnknownOpDynamicallyLegal([] (operation op) {
                return !has_hl_type(op);
            });

            mlir::RewritePatternSet patterns(&mctx);
            const auto &dl_analysis = this->getAnalysis< mlir::DataLayoutAnalysis >();
            TypeConverter type_converter(dl_analysis.getAtOrAbove(op), mctx);

            patterns.add< LowerHighLevelOpType, LowerFuncOpType >(
                type_converter, patterns.getContext()
            );

            if (mlir::failed(mlir::applyPartialConversion(op, trg, std::move(patterns)))) {
                return signalPassFailure();
            }
        }
    };

    mlir::Block &solo_block(mlir::Region &region) {
        VAST_ASSERT(region.hasOneBlock());
        return *region.begin();
    }

    // TODO(lukas):
    struct LowerStructDeclOp : mlir::OpConversionPattern< hl::StructDeclOp >
    {
        using parent_t = mlir::OpConversionPattern< hl::StructDeclOp >;

        // TODO(lukas): We most likely no longer need type converter here.
        LowerStructDeclOp(TypeConverter &tc, mlir::MLIRContext *mctx) : parent_t(tc, mctx) {}

        std::vector< mlir_type > collect_field_tys(hl::StructDeclOp op) const {
            std::vector< mlir_type > out;
            for (auto &maybe_field : solo_block(op.getFields())) {
                auto field = mlir::dyn_cast< hl::FieldDeclOp >(maybe_field);
                VAST_ASSERT(field);
                out.push_back(field.getType());
            }
            return out;
        }

        // TODO(lukas): This is definitely **not** how it should be done.
        //              Rework once links via symbols have api.
        std::vector< hl::TypeDeclOp > fetch_decls(hl::StructDeclOp op) const {
            std::vector< hl::TypeDeclOp > out;
            auto module_op = op->getParentOfType< mlir::ModuleOp >();
            for (auto &x : solo_block(module_op.getBodyRegion())) {
                if (auto type_decl = mlir::dyn_cast< hl::TypeDeclOp >(x);
                    type_decl && type_decl.getName() == op.getName())
                {
                    out.push_back(type_decl);
                }
            }
            return out;
        }

        logical_result matchAndRewrite(
            hl::StructDeclOp op, hl::StructDeclOp::Adaptor ops,
            conversion_rewriter &rewriter
        ) const override {
            auto field_tys = collect_field_tys(op);
            auto trg_ty    = mlir::TupleType::get(this->getContext(), field_tys);

            rewriter.create< hl::TypeDefOp >(op.getLoc(), op.getName(), trg_ty);

            auto type_decls = fetch_decls(op);
            for (auto x : type_decls) {
                rewriter.eraseOp(x);
            }

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

        auto _illegal() {
            return [&]< typename O >() { trg.addIllegalOp< O >(); };
        }

        template< typename O, typename... Os, typename Fn >
        self_t &_rec(Fn &&fn) {
            fn.template operator()< O >();
            if constexpr (sizeof...(Os) == 0) {
                return *this;
            } else {
                return _rec< Os... >(std::forward< Fn >(fn));
            }
        }

        template< typename O, typename... Os >
        self_t &illegal() {
            return _rec< O, Os... >(_illegal());
        }

        self_t &unkown_as_legal() {
            trg.markUnknownOpDynamicallyLegal([](auto) { return true; });
            return *this;
        }
    };
} // namespace vast::hl

std::unique_ptr< mlir::Pass > vast::hl::createHLLowerTypesPass() {
    return std::make_unique< HLLowerTypesPass >();
}
