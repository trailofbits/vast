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

#include <iostream>

namespace vast::hl
{
    bool contains_hl_type(mlir::Type t)
    {
        CHECK(static_cast< bool >(t), "Argument of in `contains_hl_type` is not valid.");
        // We need to manually check `t` itself.
        bool found = t.isa< hl::HighLevelType >();
        auto is_hl = [&](auto t)
        {
            if (t.template isa< hl::RecordType >() || t.template isa< hl::ArrayType >())
                return;
            found |= t.template isa< hl::HighLevelType >();
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


    auto get_typeattr(mlir::Operation *op, const std::string &key = "type")
    {
        return op->template getAttrOfType< mlir::TypeAttr >(key);
    }

    template< typename Trg >
    bool has_typeattr_of(mlir::Operation *op)
    {
        if (auto x = get_typeattr(op))
            return contains_hl_type(x.getValue());
        return false;
    }

    bool has_hl_typeattr(mlir::Operation *op) {
        return has_typeattr_of< hl::HighLevelType >(op);
    }

    bool has_hl_type(mlir::Operation *op)
    {
        return contain_hl_type(op->getResultTypes()) ||
               contain_hl_type(op->getOperandTypes()) ||
               has_hl_typeattr(op);
    }

    bool should_lower(mlir::Operation *op) { return !has_hl_type(op); }

    struct TypeConverter : mlir::TypeConverter {
        const mlir::DataLayout &dl;
        mlir::MLIRContext &mctx;

        TypeConverter(const mlir::DataLayout &dl_, mlir::MLIRContext &mctx_)
            : mlir::TypeConverter(), dl(dl_), mctx(mctx_)
        {
            // Fallthrough option - we define it first as it seems the framework
            // goes from the last added conversion.
            addConversion([&](mlir::Type t) -> llvm::Optional< mlir::Type > {
                if (!t.dyn_cast< hl::HighLevelType >())
                    return t;
                return llvm::None;
            });
            // Use provided data layout to get the correct type.
            addConversion([&](hl::IntegerType t) { return this->convert_int_t(t); });
            addConversion([&](hl::BoolType t) { return this->convert_bool_t(t); });
            addConversion([&](hl::PointerType t) { return this->convert_ptr_t(t); });
            addConversion([&](mlir::FunctionType t) { return this->convert_fn_t(t); });
        }

        using types_t = mlir::SmallVector< mlir::Type >;
        using maybe_types_t = std::optional< types_t >;

        maybe_types_t convert_type(mlir::Type t)
        {
            types_t out;
            if (mlir::succeeded(convertTypes(t, out)))
                return { std::move( out ) };
            return {};
        }

        using maybe_t = llvm::Optional< mlir::Type >;
        maybe_t convert_int_t(hl::IntegerType t)
        {
            return mlir::IntegerType::get(&mctx, dl.getTypeSizeInBits(t));
        }

        maybe_t convert_bool_t(hl::BoolType t)
        {
            return mlir::IntegerType::get(&mctx, dl.getTypeSizeInBits(t));
        }

        maybe_t convert_ptr_t(hl::PointerType t)
        {
            if (auto nested_tys = convert_type(t.getElementType());
                nested_tys && nested_tys->size() == 1)
            {
                // TODO(lukas): Address spaces.
                // TODO(lukas): Not sure if other than UnrankedMemRef can be constructed
                //              generically.
                return mlir::UnrankedMemRefType::get(*(nested_tys->begin()), 0);
            }
            return llvm::None;
        }


        maybe_t convert_fn_t(mlir::FunctionType t)
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

    // `ConversionPattern` provides methods that can use `TypeConverter`, which
    // other patterns do not.
    struct LowerHLTypePattern : mlir::ConversionPattern
    {
        LowerHLTypePattern(TypeConverter &tc, mlir::MLIRContext *mctx)
            : mlir::ConversionPattern(tc, mlir::Pattern::MatchAnyOpTypeTag{}, 1, mctx)
        {}

        // `ops` are remapped operands.
        // `op` is current operation (with old operands).
        // `rewriter` is created by mlir when we start conversion.
        mlir::LogicalResult matchAndRewrite(
                mlir::Operation *op, mlir::ArrayRef< mlir::Value > ops,
                mlir::ConversionPatternRewriter &rewriter) const override
        {
            mlir::SmallVector< mlir::Type > rty;
            auto status = this->getTypeConverter()->convertTypes(op->getResultTypes(), rty);
            assert(mlir::succeeded(status));

            // We just change type, no need to copy everything
            auto lower_op = [&]() {
                assert(rty.size() == op->getResults().size());
                for (std::size_t i = 0; i < op->getResults().size(); ++i)
                    op->getResult(i).setType(rty[i]);

                // TODO(lukas): Investigate if moving to separate pattern is better
                //              way to do this.
                // TODO(lukas): Other operations that can have block arguments.
                if (auto fn = mlir::dyn_cast_or_null< mlir::FuncOp >(op))
                    if (mlir::failed(rewriter.convertRegionTypes(&fn.getBody(),
                                                                 *getTypeConverter())))
                        assert(false && "Cannot handle failure to update block types.");
                // For example return type of function can be encoded in attributes
                lower_attrs(op);
            };
            // It has to be done in one "transaction".
            rewriter.updateRootInPlace(op, lower_op);

            return mlir::success();
        }

        void lower_attrs(mlir::Operation *op) const
        {
            auto attr = op->getAttrDictionary().getNamed("type");
            if (!attr.hasValue())
                return;

            auto as_t = attr->second.dyn_cast< mlir::TypeAttr >();
            assert(as_t);
            auto converted = getTypeConverter()->convertType(as_t.getValue());
            op->setAttr("type", mlir::TypeAttr::get(converted));
        }
    };

    struct LowerHighLevelTypesPass : LowerHighLevelTypesBase< LowerHighLevelTypesPass >
    {
        void runOnOperation() override;
    };

    void LowerHighLevelTypesPass::runOnOperation()
    {
        mlir::ConversionTarget trg(this->getContext());
        // We want to check *everything* for presence of hl type
        // that can be lowered.
        trg.markUnknownOpDynamicallyLegal(should_lower);

        mlir::RewritePatternSet patterns(&this->getContext());
        // TODO(lukas): This is expensive, construct once per module.
        const auto &dl_analysis = this->getAnalysis< mlir::DataLayoutAnalysis >();
        TypeConverter type_converter(dl_analysis.getAtOrAbove(this->getOperation()),
                                     this->getContext());
        patterns.add< LowerHLTypePattern >(type_converter, patterns.getContext());

        if (mlir::failed(mlir::applyPartialConversion(
                        this->getOperation(),trg, std::move(patterns))))
            return signalPassFailure();
    }
}


std::unique_ptr< mlir::Pass > vast::hl::createLowerHighLevelTypesPass()
{
  return std::make_unique< LowerHighLevelTypesPass >();
}
