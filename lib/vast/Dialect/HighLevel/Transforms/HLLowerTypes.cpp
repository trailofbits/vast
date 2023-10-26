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
#include "vast/Util/TypeUtils.hpp"

#include "vast/Conversion/TypeConverters/HLToStd.hpp"
#include "vast/Conversion/TypeConverters/TypeConverter.hpp"

#include <algorithm>
#include <iostream>

namespace vast::hl
{
    using type_converter_t = conv::tc::HLToStd;

    struct LowerHighLevelOpType : mlir::ConversionPattern
    {
        using Base = mlir::ConversionPattern;
        using Base::Base;

        LowerHighLevelOpType(type_converter_t &tc, mcontext_t *mctx)
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

            auto &tc = static_cast< type_converter_t & >(*getTypeConverter());

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
                replacer.addReplacement(conv::tc::convert_type_attr(tc));
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
            auto &tc = static_cast< type_converter_t & >(*getTypeConverter());

            conv::tc::signature_conversion_t sigconvert(fty.getNumInputs());
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
            auto is_legal = [](operation op)
            {
                auto is_hl = [](mlir_type t) -> bool { return isHighLevelType(t); };

                return !has_type_somewhere(op, is_hl);
            };
            trg.markUnknownOpDynamicallyLegal(is_legal);

            mlir::RewritePatternSet patterns(&mctx);
            const auto &dl_analysis = this->getAnalysis< mlir::DataLayoutAnalysis >();
            type_converter_t type_converter(dl_analysis.getAtOrAbove(op), mctx);

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
        LowerStructDeclOp(type_converter_t &tc, mlir::MLIRContext *mctx) : parent_t(tc, mctx) {}

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
