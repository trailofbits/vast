// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Dialect/HighLevel/HighLevelOps.hpp"
#include "vast/Translation/Context.hpp"
#include "vast/Util/Warnings.hpp"

namespace vast::hl
{
    struct HighLevelBuilder {
        using InsertPoint = mlir::OpBuilder::InsertPoint;

        HighLevelBuilder(TranslationContext &ctx)
            : ctx(ctx)
            , builder(ctx.getBodyRegion()) {}

        mlir::Location get_location(clang::SourceRange range) {
            return get_location_impl(range.getBegin());
        }

        mlir::Location get_end_location(clang::SourceRange range) {
            return get_location_impl(range.getEnd());
        }

        template< typename Op, typename... Args >
        auto make(Args &&...args) {
            auto convert = overloaded{ to_value, identity };
            return builder.create< Op >(convert(std::forward< Args >(args))...);
        }

        template< typename Op, typename... Args >
        Value make_value(Args &&...args) {
            return make< Op >(std::forward< Args >(args)...);
        }

        template< typename Op, typename... Args >
        Stmt make_stmt(Args &&...args) {
            return make< Op >(std::forward< Args >(args)...);
        }

        InsertPoint save_insertion_point() { return builder.saveInsertionPoint(); }
        void restore_insertion_point(InsertPoint ip) { builder.restoreInsertionPoint(ip); }

        void set_insertion_point_to_start(mlir::Block *block) {
            builder.setInsertionPointToStart(block);
        }

        void set_insertion_point_to_end(mlir::Block *block) {
            builder.setInsertionPointToEnd(block);
        }

        mlir::Block *get_block() const { return builder.getBlock(); }

        mlir::Block *create_block(mlir::Region *parent) { return builder.createBlock(parent); }

        BoolType bool_type() { return BoolType::get(&ctx.getMLIRContext()); }

        mlir::Value bool_value(mlir::Location loc, bool value) {
            auto attr = mlir::BoolAttr::get(&ctx.getMLIRContext(), value);
            return make< ConstantIntOp >(loc, bool_type(), attr);
        }

        mlir::Value true_value(mlir::Location loc) { return bool_value(loc, true); }
        mlir::Value false_value(mlir::Location loc) { return bool_value(loc, false); }

        mlir::Value constant(mlir::Location loc, mlir::Type ty, bool value) {
            CHECK(ty.isa< BoolType >(), "mismatched boolean constant type");
            return bool_value(loc, value);
        }

        mlir::Value constant(mlir::Location loc, mlir::Type ty, llvm::APInt value) {
            return make< ConstantIntOp >(loc, ty, value);
        }

        mlir::Value constant(mlir::Location loc, mlir::Type ty, llvm::APSInt value) {
            return make< ConstantIntOp >(loc, ty, value);
        }

        mlir::Value constant(mlir::Location loc, mlir::Type ty, unsigned int value) {
            return constant(loc, ty, llvm::APInt(32, value));
        }

        mlir::Value constant(mlir::Location loc, mlir::Type ty, llvm::APFloat value) {
            return make< ConstantFloatOp >(loc, ty, value);
        }

        mlir::Value constant(mlir::Location loc, mlir::Type ty, llvm::StringRef value) {
            CHECK(ty.isa< ConstantArrayType >(), "string constant must have array type");
            auto attr = mlir::StringAttr::get(value, ty);
            return make< ConstantStringOp >(loc, ty.cast< ConstantArrayType >(), attr);
        }

      private:
        mlir::Location get_location_impl(clang::SourceLocation at) {
            auto loc = ctx.getSourceManager().getPresumedLoc(at);

            if (loc.isInvalid())
                return builder.getUnknownLoc();

            auto file = mlir::Identifier::get(loc.getFilename(), &ctx.getMLIRContext());
            return mlir::FileLineColLoc::get(file, loc.getLine(), loc.getColumn());
        }

        TranslationContext &ctx;
        mlir::OpBuilder builder;
    };

    struct ScopedInsertPoint {
        using InsertPoint = mlir::OpBuilder::InsertPoint;

        ScopedInsertPoint(HighLevelBuilder &builder)
            : builder(builder)
            , point(builder.save_insertion_point()) {}

        ~ScopedInsertPoint() { builder.restore_insertion_point(point); }

        HighLevelBuilder &builder;
        InsertPoint point;
    };
} // namespace vast::hl
