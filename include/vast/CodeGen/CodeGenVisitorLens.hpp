// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <clang/AST/DeclVisitor.h>
VAST_UNRELAX_WARNINGS

#include "vast/CodeGen/CodeGenMeta.hpp"
#include "vast/CodeGen/CodeGenVisitorBase.hpp"
#include "vast/CodeGen/Mangler.hpp"
#include "vast/CodeGen/Util.hpp"

namespace vast::cg {

    //
    // visitor_lens
    //
    // Allows to access the visitor base from mixins.
    //
    template< typename derived_t, template< typename > class base_t >
    struct visitor_lens
    {
        friend base_t< derived_t >;

        auto derived()       -> derived_t      & { return static_cast<derived_t&>(*this); }
        auto derived() const -> derived_t const& { return static_cast<derived_t const&>(*this); }

        //
        // Contexts
        //
        auto context()       -> auto      & { return derived().ctx; }
        auto context() const -> auto const& { return derived().ctx; }

        auto mcontext()       -> mcontext_t      & { return context().mctx; }
        auto mcontext() const -> mcontext_t const& { return context().mctx; }

        auto acontext()       -> acontext_t      & { return context().actx; }
        auto acontext() const -> acontext_t const& { return context().actx; }

        auto name_mangler()       -> mangler_t      & { return context().mangler; }
        auto name_mangler() const -> mangler_t const& { return context().mangler; }

        //
        // builder
        //
        decltype(auto) mlir_builder() {
            return derived().mlir_builder();
        }

        decltype(auto) make_value_builder(const clang::Stmt *stmt) {
            return derived().make_value_builder(stmt);
        }

        decltype(auto) insertion_guard() {
            return derived().insertion_guard();
        }

        decltype(auto) make_yield_true() { return derived().make_yield_true(); }


        decltype(auto) make_cond_builder(const clang::Stmt *stmt) {
            return derived().make_cond_builder(stmt);
        }

        decltype(auto) make_region_builder(const clang::Stmt *stmt) {
            return derived().make_region_builder(stmt);
        }

        decltype(auto) make_value_yield_region(auto stmt) {
            return derived().make_value_yield_region(stmt);
        }

        decltype(auto) make_stmt_expr_region(const clang::CompoundStmt *stmt) {
            return derived().make_stmt_expr_region(stmt);
        }

        decltype(auto) set_insertion_point_to_start(region_ptr region) {
            return derived().set_insertion_point_to_start(region);
        }

        decltype(auto) set_insertion_point_to_end(region_ptr region) {
            return derived().set_insertion_point_to_end(region);
        }

        decltype(auto) set_insertion_point_to_start(block_ptr block) {
            return derived().set_insertion_point_to_start(block);
        }

        decltype(auto) set_insertion_point_to_end(block_ptr block) {
            return derived().set_insertion_point_to_end(block);
        }

        bool has_insertion_block() {
            return derived().has_insertion_block();
        }

        template< typename... args_t >
        decltype(auto) constant(args_t &&...args) {
            return derived().constant(std::forward< args_t >(args)...);
        }

        template< typename Token >
        decltype(auto) visit(Token token) { return derived().Visit(token); }

        template< typename Token >
        mlir_type visit_as_lvalue_type(Token token) { return derived().VisitLValueType(token); }

        decltype(auto) visit_function_type(const clang::FunctionType *fty, bool variadic) {
            return derived().VisitCoreFunctionType(fty, variadic);
        }

        template< typename op_t >
        decltype(auto) make_operation() {
            return derived().template make_operation< op_t >();
        }

        decltype(auto) make_type_yield_builder(const clang::Expr *expr) {
            return derived().make_type_yield_builder(expr);
        }

        template< typename Token >
        loc_t meta_location(Token token) const {
            return derived().meta_location(token);
        }
    };

} // namespace vast::cg
