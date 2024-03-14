// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

#include "vast/Util/Common.hpp"
#include "vast/CodeGen/CodeGenVisitorLens.hpp"

#include "vast/Dialect/Core/CoreOps.hpp"
#include "vast/Dialect/Core/CoreAttributes.hpp"

#include "vast/Dialect/HighLevel/HighLevelOps.hpp"
#include "vast/Dialect/HighLevel/HighLevelAttributes.hpp"
#include "vast/Dialect/HighLevel/HighLevelTypes.hpp"

namespace vast::cg {

    template< typename scope_t >
    struct scope_generator_guard : insertion_guard {
        scope_generator_guard(mlir_builder &builder, loc_t loc)
            : insertion_guard(builder), loc(loc)
            , scope(builder.create< scope_t >(loc))
        {
            auto &block = scope.getBody().emplaceBlock();
            builder.setInsertionPointToStart(&block);
        }

        scope_t get() { return scope; }

        loc_t loc;
        scope_t scope;
    };

    using CoreScope = scope_generator_guard< core::ScopeOp >;
    using TranslationUnitScope = scope_generator_guard< hl::TranslationUnitOp >;

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
    // builder_t
    //
    // Allows to create new nodes from within mixins.
    //
    template< typename derived_t >
    struct builder_t : visitor_lens< derived_t, builder_t >
    {
        using lens = visitor_lens< derived_t, builder_t >;

        using lens::mlir_builder;
        using lens::acontext;

        using lens::visit;

        using lens::meta_location;

        template< typename op_t >
        op_t get_parent_of_type() {
            auto reg = mlir_builder().getInsertionBlock()->getParent();
            return reg->template getParentOfType< op_t >();
        }

        void set_insertion_point_to_start(region_ptr region) {
            mlir_builder().setInsertionPointToStart(&region->front());
        }

        void set_insertion_point_to_end(region_ptr region) {
            mlir_builder().setInsertionPointToEnd(&region->back());
        }

        void set_insertion_point_to_start(block_ptr block) {
            mlir_builder().setInsertionPointToStart(block);
        }

        void set_insertion_point_to_end(block_ptr block) {
            mlir_builder().setInsertionPointToEnd(block);
        }

        bool has_insertion_block() {
            return mlir_builder().getInsertionBlock();
        }

        void clear_insertion_point() {
            mlir_builder().clearInsertionPoint();
        }

        insertion_guard insertion_guard() {
            return { mlir_builder() };
        }


        template< typename op_t, typename... args_t >
        auto create(args_t &&...args) {
            return mlir_builder().template create< op_t >(std::forward< args_t >(args)...);
        }

        template< typename result_type, typename builder_type >
        auto make_compose_impl(builder_type &&builder) {
            return compose_state_t< result_type, builder_type >(std::forward< builder_type >(builder));
        }

        template< typename op_t >
        auto make_operation() {
            return make_compose_impl< op_t >([&] (auto&& ...args) {
                return create< op_t >(std::forward< decltype(args) >(args)...);
            });
        }

        template< typename type >
        auto make_type() {
            return make_compose_impl< type >([&] (auto&& ...args) {
                return type::get(std::forward< decltype(args) >(args)...);
            });
        }

        template< typename scope_t >
        auto make_scoped(loc_t loc, auto content_builder) {
            scope_t scope(mlir_builder(), loc);
            content_builder();
            return scope.get();
        }

        // TODO: unify region and stmt builders, they are the same thing but
        // different region builders directly emit new region, while stmt
        // builders return callback to emit region later.
        //
        // We want to use just region builders in the future.

        std::unique_ptr< region_t > make_empty_region() {
            auto reg = std::make_unique< region_t >();
            reg->emplaceBlock();
            return reg;
        }

        std::unique_ptr< region_t > make_stmt_region(const clang::Stmt *stmt) {
            auto guard = insertion_guard();

            auto reg = make_empty_region();
            set_insertion_point_to_start( reg.get() );
            visit(stmt);

            return reg;
        }

        std::unique_ptr< region_t > make_stmt_region(const clang::CompoundStmt *stmt) {
            auto guard = insertion_guard();

            auto reg = make_empty_region();
            set_insertion_point_to_start( reg.get() );
            for (auto s : stmt->body()) {
                visit(s);
            }

            return reg;
        }

        using reg_and_type = std::pair< std::unique_ptr< region_t >, mlir_type >;

        template< typename stmt_t >
        reg_and_type make_value_yield_region(const stmt_t *stmt) {
            auto guard  = insertion_guard();
            auto reg    = make_stmt_region(stmt);

            auto &block = reg->back();
            set_insertion_point_to_end( &block );
            auto type = block.back().getResult(0).getType();
            VAST_CHECK(block.back().getNumResults(), "value region require last operation to be value");
            create< hl::ValueYieldOp >(meta_location(stmt), block.back().getResult(0));

            return { std::move(reg), type };
        }

        hl::VoidType void_type() {
            return visit(acontext().VoidTy).template cast< hl::VoidType >();
        }

        mlir_value void_value(loc_t loc) {
            return create< hl::ConstantOp >(loc, void_type());
        }

        reg_and_type make_stmt_expr_region (const clang::CompoundStmt *stmt) {
            auto guard  = insertion_guard();
            auto reg    = make_stmt_region(stmt);

            auto &block = reg->back();
            set_insertion_point_to_end( &block );

            // ({5;;;;;}) <- this is supposed to return 5...
            auto last = std::prev(block.end());
            while (last != block.begin() && isa< hl::SkipStmt >(&*last)) {
                last = std::prev(last);
            }

            auto loc = meta_location(stmt);

            if (last->getNumResults() > 0) {
                auto res = last->getResult(0);
                create< hl::ValueYieldOp >(loc, res);
                return { std::move(reg), res.getType()};
            }

            auto void_const = void_value(loc);
            create< hl::ValueYieldOp >(meta_location(stmt), void_const);
            return { std::move(reg), void_const.getType()};
        }

        template< typename yield_t >
        auto make_stmt_builder(const clang::Stmt *stmt) {
            return [stmt, this](auto &bld, auto loc) {
                visit(stmt);
                auto &op = bld.getBlock()->back();
                VAST_ASSERT(op.getNumResults() <= 1);
                if (op.getNumResults() > 0)
                    create< yield_t >(loc, op.getResult(0));
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

        mlir_value bool_value(loc_t loc, bool value) {
            return create< hl::ConstantOp >(loc, bool_type(), value);
        }

        mlir_value true_value(loc_t loc)  { return bool_value(loc, true); }
        mlir_value false_value(loc_t loc) { return bool_value(loc, false); }

        mlir_value constant(loc_t loc) {
            return void_value(loc);
        }
        mlir_value constant(loc_t loc, bool value) {
            return bool_value(loc, value);
        }

        mlir_value constant(loc_t loc, mlir_type ty, ap_int value) {
            return create< hl::ConstantOp >(loc, ty, value);
        }

        mlir_value constant(loc_t loc, mlir_type ty, ap_sint value) {
            return create< hl::ConstantOp >(loc, ty, value);
        }

        mlir_value constant(loc_t loc, mlir_type ty, ap_float value) {
            return create< hl::ConstantOp >(loc, ty, value);
        }

        mlir_value constant(loc_t loc, mlir_type ty, string_ref value) {
            return create< hl::ConstantOp >(loc, ty, value);
        }
    };

} // namespace vast::cg
