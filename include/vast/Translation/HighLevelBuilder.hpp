// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <clang/Basic/SourceManager.h>
VAST_UNRELAX_WARNINGS

#include "vast/Dialect/HighLevel/HighLevelOps.hpp"
#include "vast/Translation/Context.hpp"
#include "vast/Translation/HighLevelTypeConverter.hpp"

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

        void set_insertion_point_to_start(mlir::Region *region) {
            builder.setInsertionPointToStart(&region->front());
        }

        void set_insertion_point_to_end(mlir::Region *region) {
            builder.setInsertionPointToEnd(&region->back());
        }

        void set_insertion_point_to_start(mlir::Block *block) {
            builder.setInsertionPointToStart(block);
        }

        void set_insertion_point_to_end(mlir::Block *block) {
            builder.setInsertionPointToEnd(block);
        }

        mlir::Block *get_block() const { return builder.getBlock(); }

        mlir::Block *create_block(mlir::Region *parent) { return builder.createBlock(parent); }

        BoolType bool_type() {
            // We go via `types.convert` since it may need to process the type
            // in order to emit data layout.
            HighLevelTypeConverter types(ctx);
            auto &actx = ctx.getASTContext();
            return types.convert(actx.BoolTy).cast< BoolType >();
        }

        mlir::Value bool_value(mlir::Location loc, bool value) {
            auto attr = mlir::BoolAttr::get(&ctx.getMLIRContext(), value);
            return make< ConstantIntOp >(loc, bool_type(), attr);
        }

        mlir::Value true_value(mlir::Location loc) { return bool_value(loc, true); }
        mlir::Value false_value(mlir::Location loc) { return bool_value(loc, false); }

        mlir::Value constant(mlir::Location loc, mlir::Type ty, bool value) {
            VAST_CHECK(ty.isa< BoolType >(), "mismatched boolean constant type");
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
            auto attr = builder.getFloatAttr(to_std_float_type(ty), value);
            return make< ConstantFloatOp >(loc, ty, attr);
        }

        mlir::Value constant(mlir::Location loc, mlir::Type ty, llvm::StringRef value) {
            VAST_CHECK(ty.isa< ArrayType >(), "string constant must have array type");
            auto attr = mlir::StringAttr::get(value, ty);
            return make< ConstantStringOp >(loc, ty.cast< ArrayType >(), attr);
        }

        auto declare_type(mlir::Location loc, llvm::StringRef name) {
            if (auto decl = ctx.type_decls.lookup(name)) {
                return decl;
            }

            auto decl = make< TypeDeclOp >(loc, name);
            if (failed(ctx.type_decls.declare(name, decl))) {
                ctx.error("error: multiple type declarations with the same name");
            }
            return decl;
        }

        auto define_type(mlir::Location loc, mlir::Type type, llvm::StringRef name) {
            if (auto def = ctx.type_defs.lookup(name))
                return def;

            auto def = make< TypeDefOp >(loc, name, type);
            if (failed(ctx.type_defs.declare(name, def))) {
                ctx.error("error: multiple type definitions with the same name");
            }

            return def;
        }

        auto declare_enum(mlir::Location loc, llvm::StringRef name, Type type, BuilderCallback constants) {
            auto decl = make< EnumDeclOp >(loc, name, type, constants);
            if (failed(ctx.enum_decls.declare(name, decl))) {
                ctx.error("error: multiple enum declarations with the same name");
            }

            return decl;
        }

        auto declare_enum_constant(EnumConstantOp enum_constant) {
            auto name = enum_constant.name();
            if (auto decl = ctx.enum_constants.lookup(name))
                return decl;

            if (failed(ctx.enum_constants.declare(name, enum_constant))) {
                ctx.error("error: multiple enum constant declarations with the same name");
            }

            return enum_constant;
        }

        mlir::OpBuilder& get_op_builder() { return builder; }

      private:
        mlir::Location get_location_impl(clang::SourceLocation at) {
            auto loc = ctx.getSourceManager().getPresumedLoc(at);

            if (loc.isInvalid())
                return builder.getUnknownLoc();

            return mlir::FileLineColLoc::get(&ctx.getMLIRContext(), loc.getFilename(), loc.getLine(), loc.getColumn());
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

    struct HighLevelScope : ScopedInsertPoint {
        HighLevelScope(HighLevelBuilder &builder, Location loc)
            : ScopedInsertPoint(builder), loc(loc)
        {
            scope = builder.make< ScopeOp >(loc);
            // TODO(Heno): move to scope ctor & use builder
            auto &body    = scope.body();
            body.push_back(new mlir::Block());
            builder.set_insertion_point_to_start(&body.front());
        }

        ScopeOp get() { return scope; }

        ScopeOp scope;
        Location loc;
    };

    auto make_scoped(HighLevelBuilder &builder, Location loc, auto content_builder) {
        HighLevelScope scope(builder, loc);
        content_builder();
        return scope.get();
    }

} // namespace vast::hl
