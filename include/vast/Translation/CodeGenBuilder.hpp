// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

#include "vast/Translation/CodeGenVisitorLens.hpp"

namespace vast::hl {

    using InsertionGuard = mlir::OpBuilder::InsertionGuard;

    struct CodeGenBuilderHandle {
        mlir::OpBuilder &builder;
    };

    template< typename Scope >
    struct ScopeGenerator : InsertionGuard {
        ScopeGenerator(CodeGenBuilderHandle handle, Location loc)
            : InsertionGuard(handle.builder), loc(loc)
            , scope(handle.builder.create< Scope >(loc))
        {
            auto &block = scope.body().emplaceBlock();
            handle.builder.setInsertionPointToStart(&block);
        }

        Scope get() { return scope; }

        Location loc;
        Scope scope;
    };

    using HighLevelScope = ScopeGenerator< ScopeOp >;
    using TranslationUnitScope = ScopeGenerator< TranslationUnitOp >;

    //
    // Operation composable builder
    //
    template< typename Op >
    struct OperationState;

    template< typename Op >
    OperationState(Op) -> OperationState< Op >;

    template< typename Op >
    struct OperationState {
        OperationState(Op &&op) : op(std::move(op)) {}

        template< typename arg_t >
        constexpr auto bind(arg_t &&arg) && {
            auto binded = [arg = std::forward< arg_t >(arg), op = std::move(op)] (auto &&...args) {
                return op(arg, std::forward< decltype(args) >(args)...);
            };
            return OperationState< decltype(binded) >(std::move(binded));
        }

        template< typename arg_t >
        constexpr auto bind_if(bool cond, arg_t &&arg) && {
            auto binded = [cond, arg = std::forward< arg_t >(arg), op = std::move(op)] (auto &&...args) {
                if (cond)
                    return op(arg, std::forward< decltype(args) >(args)...);
                return op(std::forward< decltype(args) >(args)...);
            };
            return OperationState< decltype(binded) >(std::move(binded));
        }

        template< typename arg_t >
        constexpr auto bind_region_if(bool cond, arg_t &&arg) && {
            auto binded = [cond, arg = std::forward< arg_t >(arg), op = std::move(op)] (auto &&...args) {
                if (cond)
                    return op(arg, std::forward< decltype(args) >(args)...);
                return op(std::nullopt, std::forward< decltype(args) >(args)...);
            };
            return OperationState< decltype(binded) >(std::move(binded));
        }

        auto freeze() { return op(); }

        Op op;
    };

    //
    // CodeGenBuilder
    //
    // Allows to create new nodes from within mixins.
    //
    template< typename Base, typename Derived >
    struct CodeGenBuilderMixin
        : CodeGenVisitorLens< CodeGenBuilderMixin< Base, Derived >, Derived >
    {
        using LensType = CodeGenVisitorLens< CodeGenBuilderMixin< Base, Derived >, Derived >;

        using LensType::derived;
        using LensType::context;
        using LensType::mcontext;
        using LensType::acontext;

        using LensType::meta_location;

        using LensType::visit;

        auto op_builder() -> mlir::OpBuilder & { return derived()._builder; }

        auto builder() -> CodeGenBuilderHandle { return { op_builder() }; }

        void set_insertion_point_to_start(mlir::Region *region) {
            op_builder().setInsertionPointToStart(&region->front());
        }

        void set_insertion_point_to_end(mlir::Region *region) {
            op_builder().setInsertionPointToEnd(&region->back());
        }

        void set_insertion_point_to_start(mlir::Block *block) {
            op_builder().setInsertionPointToStart(block);
        }

        void set_insertion_point_to_end(mlir::Block *block) {
            op_builder().setInsertionPointToEnd(block);
        }

        template< typename Op, typename... Args >
        auto create(Args &&...args) {
            return op_builder().template create< Op >(std::forward< Args >(args)...);
        }

        template< typename op >
        auto make_operation() {
            return OperationState([&] (auto&& ...args) {
                return create< op >(std::forward< decltype(args) >(args)...);
            });
        }

        template< typename Scope >
        auto make_scoped(Location loc, auto content_builder) {
            Scope scope(builder(), loc);
            content_builder();
            return scope.get();
        }

        InsertionGuard insertion_guard() { return InsertionGuard(op_builder()); }

        // TODO: unify region and stmt builders, they are the same thing but
        // different region builders directly emit new region, while stmt
        // builders return callback to emit region later.
        //
        // We want to use just region builders in the future.

        std::unique_ptr< Region > make_stmt_region(const clang::Stmt *stmt) {
            auto guard = insertion_guard();

            auto reg = std::make_unique< Region >();
            set_insertion_point_to_start( &reg->emplaceBlock() );
            visit(stmt);

            return reg;
        }

        using RegionAndType = std::pair< std::unique_ptr< Region >, Type >;
        RegionAndType make_value_yield_region(const clang::Stmt *stmt) {
            auto guard  = insertion_guard();
            auto reg    = make_stmt_region(stmt);

            auto &block = reg->back();
            set_insertion_point_to_end( &block );
            auto type = block.back().getResult(0).getType();
            create< ValueYieldOp >(meta_location(stmt), block.back().getResult(0));

            return { std::move(reg), type };
        }

        template< typename YieldType >
        auto make_stmt_builder(const clang::Stmt *stmt) {
            return [stmt, this](auto &bld, auto loc) {
                visit(stmt);
                auto &op = bld.getBlock()->back();
                VAST_ASSERT(op.getNumResults() == 1);
                create< YieldType >(loc, op.getResult(0));
            };
        }

        auto make_value_builder(const clang::Stmt *stmt) {
            return make_stmt_builder< ValueYieldOp >(stmt);
        }

        auto make_cond_builder(const clang::Stmt *stmt) {
            return make_stmt_builder< CondYieldOp >(stmt);
        }

        auto make_region_builder(const clang::Stmt *stmt) {
            return [stmt, this](auto &bld, auto) {
                if (stmt) visit(stmt);
                // TODO let other pass remove trailing scopes?
                splice_trailing_scopes(*bld.getBlock()->getParent());
            };
        }

        auto make_yield_true() {
            return [this](auto &bld, auto loc) {
                create< CondYieldOp >(loc, true_value(loc));
            };
        }

        BoolType bool_type() {
            return visit(acontext().BoolTy).template cast< BoolType >();
        }

        mlir::Value bool_value(mlir::Location loc, bool value) {
            auto attr = mlir::BoolAttr::get(&mcontext(), value);
            return create< ConstantIntOp >(loc, bool_type(), attr);
        }

        mlir::Value true_value(mlir::Location loc) { return bool_value(loc, true); }
        mlir::Value false_value(mlir::Location loc) { return bool_value(loc, false); }

        mlir::Value constant(mlir::Location loc, mlir::Type ty, bool value) {
            VAST_CHECK(ty.isa< BoolType >(), "mismatched boolean constant type");
            return bool_value(loc, value);
        }

        mlir::Value constant(mlir::Location loc, mlir::Type ty, llvm::APInt value) {
            return create< ConstantIntOp >(loc, ty, value);
        }

        mlir::Value constant(mlir::Location loc, mlir::Type ty, llvm::APSInt value) {
            return create< ConstantIntOp >(loc, ty, value);
        }

        mlir::Value constant(mlir::Location loc, mlir::Type ty, unsigned int value) {
            return constant(loc, ty, llvm::APInt(32, value));
        }

        mlir::Value constant(mlir::Location loc, mlir::Type ty, llvm::APFloat value) {
            auto attr = op_builder().getFloatAttr(to_std_float_type(ty), value);
            return create< ConstantFloatOp >(loc, ty, attr);
        }

        mlir::Value constant(mlir::Location loc, mlir::Type ty, llvm::StringRef value) {
            VAST_CHECK(ty.isa< ArrayType >(), "string constant must have array type");
            auto attr = mlir::StringAttr::get(value, ty);
            return create< ConstantStringOp >(loc, ty.cast< ArrayType >(), attr);
        }

        TypeDeclOp declare_type(mlir::Location loc, llvm::StringRef name) {
            if (auto decl = context().type_decls.lookup(name)) {
                return decl;
            }

            auto decl = create< TypeDeclOp >(loc, name);
            if (failed(context().type_decls.declare(name, decl))) {
                context().error("error: multiple type declarations with the same name");
            }
            return decl;
        }

        TypeDefOp define_type(mlir::Location loc, mlir::Type type, llvm::StringRef name) {
            if (auto def = context().type_defs.lookup(name)) {
                return def;
            }

            auto def = create< TypeDefOp >(loc, name, type);
            if (failed(context().type_defs.declare(name, def))) {
                context().error("error: multiple type definitions with the same name");
            }

            return def;
        }

        EnumDeclOp declare_enum(mlir::Location loc, llvm::StringRef name, Type type, BuilderCallback constants) {
            auto decl = create< EnumDeclOp >(loc, name, type, constants);
            if (failed(context().enum_decls.declare(name, decl))) {
                context().error("error: multiple enum declarations with the same name");
            }

            return decl;
        }

        EnumConstantOp declare_enum_constant(EnumConstantOp enum_constant) {
            auto name = enum_constant.name();
            if (auto decl = context().enum_constants.lookup(name)) {
                return decl;
            }

            if (failed(context().enum_constants.declare(name, enum_constant))) {
                context().error("error: multiple enum constant declarations with the same name");
            }

            return enum_constant;
        }
    };

} // namespace vast::hl
