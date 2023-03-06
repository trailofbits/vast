// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

#include "vast/Util/Common.hpp"
#include "vast/Translation/CodeGenVisitorLens.hpp"

namespace vast::cg {

    template< typename Scope >
    struct ScopeGenerator : InsertionGuard {
        ScopeGenerator(Builder &builder, Location loc)
            : InsertionGuard(builder), loc(loc)
            , scope(builder.create< Scope >(loc))
        {
            auto &block = scope.getBody().emplaceBlock();
            builder.setInsertionPointToStart(&block);
        }

        Scope get() { return scope; }

        Location loc;
        Scope scope;
    };

    using HighLevelScope       = ScopeGenerator< hl::ScopeOp >;
    using TranslationUnitScope = ScopeGenerator< hl::TranslationUnitOp >;

    //
    // composable builder state
    //
    template< typename T >
    struct ComposeState;

    template< typename T >
    ComposeState(T) -> ComposeState< T >;

    template< typename T >
    struct ComposeState {
        ComposeState(T &&value) : value(std::forward< T >(value)) {}

        template< typename ...args_t >
        constexpr inline auto bind(args_t &&...args) && {
            auto binded = [... args = std::forward< args_t >(args), value = std::move(value)] (auto &&...rest) {
                return value(args..., std::forward< decltype(rest) >(rest)...);
            };
            return ComposeState< decltype(binded) >(std::move(binded));
        }

        template< typename arg_t >
        constexpr inline auto bind_if(bool cond, arg_t &&arg) && {
            auto binded = [cond, arg = std::forward< arg_t >(arg), value = std::move(value)] (auto &&...args) {
                if (cond)
                    return value(arg, std::forward< decltype(args) >(args)...);
                return value(std::forward< decltype(args) >(args)...);
            };
            return ComposeState< decltype(binded) >(std::move(binded));
        }

        template< typename arg_t >
        constexpr inline auto bind_region_if(bool cond, arg_t &&arg) && {
            auto binded = [cond, arg = std::forward< arg_t >(arg), value = std::move(value)] (auto &&...args) {
                if (cond)
                    return value(arg, std::forward< decltype(args) >(args)...);
                return value(std::nullopt, std::forward< decltype(args) >(args)...);
            };
            return ComposeState< decltype(binded) >(std::move(binded));
        }

        auto freeze() { return value(); }

        T value;
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

        auto builder() -> Builder& { return derived()._builder; }

        template< typename Op >
        Op get_parent_of_type() {
            auto reg = builder().getInsertionBlock()->getParent();
            return reg->template getParentOfType< Op >();
        }

        hl::FuncOp get_current_function() { return get_parent_of_type< hl::FuncOp >(); }

        void set_insertion_point_to_start(mlir::Region *region) {
            builder().setInsertionPointToStart(&region->front());
        }

        void set_insertion_point_to_end(mlir::Region *region) {
            builder().setInsertionPointToEnd(&region->back());
        }

        void set_insertion_point_to_start(mlir::Block *block) {
            builder().setInsertionPointToStart(block);
        }

        void set_insertion_point_to_end(mlir::Block *block) {
            builder().setInsertionPointToEnd(block);
        }

        bool has_insertion_block() {
            return builder().getInsertionBlock();
        }

        void clear_insertion_point() {
            builder().clearInsertionPoint();
        }

        template< typename Op, typename... Args >
        auto create(Args &&...args) {
            return builder().template create< Op >(std::forward< Args >(args)...);
        }

        template< typename op >
        auto make_operation() {
            return ComposeState([&] (auto&& ...args) {
                return create< op >(std::forward< decltype(args) >(args)...);
            });
        }

        template< typename type >
        auto make_type() {
            return ComposeState([&] (auto&& ...args) {
                return type::get(std::forward< decltype(args) >(args)...);
            });
        }

        template< typename Scope >
        auto make_scoped(Location loc, auto content_builder) {
            Scope scope(builder(), loc);
            content_builder();
            return scope.get();
        }

        InsertionGuard insertion_guard() { return InsertionGuard(builder()); }

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

        std::unique_ptr< Region > make_stmt_region(const clang::CompoundStmt *stmt) {
            auto guard = insertion_guard();

            auto reg = std::make_unique< Region >();
            set_insertion_point_to_start( &reg->emplaceBlock() );
            for (auto s : stmt->body()) {
                visit(s);
            }

            return reg;
        }

        using RegionAndType = std::pair< std::unique_ptr< Region >, Type >;

        template< typename StmtType >
        RegionAndType make_value_yield_region(const StmtType *stmt) {
            auto guard  = insertion_guard();
            auto reg    = make_stmt_region(stmt);

            auto &block = reg->back();
            set_insertion_point_to_end( &block );
            auto type = block.back().getResult(0).getType();
            VAST_CHECK(type, "value region require last operation to be value");
            create< hl::ValueYieldOp >(meta_location(stmt), block.back().getResult(0));

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
            return make_stmt_builder< hl::ValueYieldOp >(stmt);
        }

        auto make_cond_builder(const clang::Stmt *stmt) {
            return make_stmt_builder< hl::CondYieldOp >(stmt);
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
                create< hl::CondYieldOp >(loc, true_value(loc));
            };
        }

        hl::BoolType bool_type() {
            return visit(acontext().BoolTy).template cast< hl::BoolType >();
        }

        mlir::Value bool_value(mlir::Location loc, bool value) {
            return create< hl::ConstantOp >(loc, bool_type(), value);
        }

        mlir::Value true_value(mlir::Location loc) { return bool_value(loc, true); }
        mlir::Value false_value(mlir::Location loc) { return bool_value(loc, false); }

        mlir::Value constant(mlir::Location loc, bool value) {
            return bool_value(loc, value);
        }

        mlir::Value constant(mlir::Location loc, mlir::Type ty, llvm::APInt value) {
            return create< hl::ConstantOp >(loc, ty, value);
        }

        mlir::Value constant(mlir::Location loc, mlir::Type ty, llvm::APSInt value) {
            return create< hl::ConstantOp >(loc, ty, value);
        }

        mlir::Value constant(mlir::Location loc, mlir::Type ty, llvm::APFloat value) {
            return create< hl::ConstantOp >(loc, ty, value);
        }

        mlir::Value constant(mlir::Location loc, mlir::Type ty, llvm::StringRef value) {
            VAST_CHECK(ty.isa< hl::ArrayType >(), "string constant must have array type");
            return create< hl::ConstantOp >(loc, ty.cast< hl::ArrayType >(), value);
        }

    };

} // namespace vast::cg
