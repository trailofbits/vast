// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

#include "vast/Util/Common.hpp"
#include "vast/CodeGen/CodeGenVisitorLens.hpp"

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
    template< typename result_type, typename bind_type >
    struct compose_state_t;

    template< typename result_type, typename bind_type >
    struct compose_state_t {

        compose_state_t(bind_type &&binder) : binder(std::forward< bind_type >(binder)) {}

        template< typename arg_t >
        static constexpr bool valid(const arg_t &arg) {
            if constexpr (std::convertible_to< arg_t , bool >) {
                return static_cast< bool >(arg);
            } else {
                // initialized non-boolean arg is always valid
                return true;
            }
        }

        template< typename ...args_t >
        static constexpr bool valid(const args_t &...args) { return (valid(args) && ...); }

        template< typename ...args_t >
        constexpr inline auto bind(args_t &&...args) && {
            auto binded = [... args = std::forward< args_t >(args), binder = std::move(binder)] (auto &&...rest) {
                if (!valid(args...)) {
                    return result_type{};
                }
                return binder(args..., std::forward< decltype(rest) >(rest)...);
            };
            return compose_state_t< result_type, decltype(binded) >(std::move(binded));
        }

        template< typename arg_t >
        constexpr inline auto bind_if(bool cond, arg_t &&arg) && {
            auto binded = [cond, arg = std::forward< arg_t >(arg), binder = std::move(binder)] (auto &&...args) {
                if (cond) {
                    if (!valid(arg)) {
                        return result_type{};
                    }

                    return binder(arg, std::forward< decltype(args) >(args)...);
                }

                return binder(std::forward< decltype(args) >(args)...);
            };
            return compose_state_t< result_type, decltype(binded) >(std::move(binded));
        }

        template< typename arg_t >
        constexpr inline auto bind_region_if(bool cond, arg_t &&arg) && {
            auto binded = [cond, arg = std::forward< arg_t >(arg), binder = std::move(binder)] (auto &&...args) {
                if (cond) {
                    if (!valid(arg)) {
                        return result_type{};
                    }

                    return binder(arg, std::forward< decltype(args) >(args)...);
                }
                return binder(std::nullopt, std::forward< decltype(args) >(args)...);
            };
            return compose_state_t< result_type, decltype(binded) >(std::move(binded));
        }

        auto freeze() { return binder(); }

        bind_type binder;
    };

    //
    // CodeGenBuilder
    //
    // Allows to create new nodes from within mixins.
    //
    template< typename Base, typename Derived >
    struct CodeGenBuilder
        : CodeGenVisitorLens< CodeGenBuilder< Base, Derived >, Derived >
    {
        using LensType = CodeGenVisitorLens< CodeGenBuilder< Base, Derived >, Derived >;

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

        template< typename result_type, typename builder_type >
        auto make_compose_impl(builder_type &&builder) {
            return compose_state_t< result_type, builder_type >(std::forward< builder_type >(builder));
        }

        template< typename op >
        auto make_operation() {
            return make_compose_impl< op >([&] (auto&& ...args) {
                return create< op >(std::forward< decltype(args) >(args)...);
            });
        }

        template< typename type >
        auto make_type() {
            return make_compose_impl< type >([&] (auto&& ...args) {
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

        std::unique_ptr< Region > make_empty_region() {
            auto reg = std::make_unique< Region >();
            reg->emplaceBlock();
            return reg;
        }

        std::unique_ptr< Region > make_stmt_region(const clang::Stmt *stmt) {
            auto guard = insertion_guard();

            auto reg = make_empty_region();
            set_insertion_point_to_start( reg.get() );
            visit(stmt);

            return reg;
        }

        std::unique_ptr< Region > make_stmt_region(const clang::CompoundStmt *stmt) {
            auto guard = insertion_guard();

            auto reg = make_empty_region();
            set_insertion_point_to_start( reg.get() );
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
            VAST_CHECK(block.back().getNumResults(), "value region require last operation to be value");
            create< hl::ValueYieldOp >(meta_location(stmt), block.back().getResult(0));

            return { std::move(reg), type };
        }

        // TODO(void-call): Remove this function in favor of make_value_yield_region
        template< typename StmtType >
        RegionAndType make_maybe_value_yield_region(const StmtType *stmt) {
            auto guard  = insertion_guard();
            auto reg    = make_stmt_region(stmt);

            auto &block = reg->back();
            auto type = Type();
            set_insertion_point_to_end( &block );
            if (block.back().getNumResults() > 0) {
                type = block.back().getResult(0).getType();
                create< hl::ValueYieldOp >(meta_location(stmt), block.back().getResult(0));
            }
            return { std::move(reg), type };
        }

        template< typename YieldType >
        auto make_stmt_builder(const clang::Stmt *stmt) {
            return [stmt, this](auto &bld, auto loc) {
                visit(stmt);
                auto &op = bld.getBlock()->back();
                VAST_ASSERT(op.getNumResults() <= 1);
                if (op.getNumResults() > 0)
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
            };
        }

        auto make_yield_true() {
            return [this](auto &bld, auto loc) {
                create< hl::CondYieldOp >(loc, true_value(loc));
            };
        }

        auto make_type_yield_builder(const clang::Expr *expr) {
            return [expr, this](auto &bld, auto loc) {
                visit(expr);

                auto block = bld.getBlock();
                VAST_CHECK(block->back().getNumResults(), "type region require last operation to be value");
                create< hl::TypeYieldOp >(meta_location(expr), block->back().getResult(0));
            };
        }

        hl::BoolType bool_type() {
            return visit(acontext().BoolTy).template cast< hl::BoolType >();
        }

        mlir::Value bool_value(mlir::Location loc, bool value) {
            return create< hl::ConstantOp >(loc, bool_type(), value);
        }

        mlir::Value true_value(mlir::Location loc)  { return bool_value(loc, true); }
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
            return create< hl::ConstantOp >(loc, ty, value);
        }

    };

} // namespace vast::cg
