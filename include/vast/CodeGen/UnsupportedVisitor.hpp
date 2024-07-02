// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

#include <gap/core/crtp.hpp>

#include "vast/CodeGen/CodeGenVisitorBase.hpp"

#include "vast/Dialect/Unsupported/UnsupportedDialect.hpp"
#include "vast/Dialect/Unsupported/UnsupportedOps.hpp"
#include "vast/Dialect/Unsupported/UnsupportedTypes.hpp"
#include "vast/Dialect/Unsupported/UnsupportedAttributes.hpp"

#include "vast/CodeGen/CodeGenBuilder.hpp"

namespace vast::cg
{
    std::string decl_name(const clang_decl *decl);

    template< typename derived >
    struct unsup_decl_visitor : gap::core::crtp< derived, unsup_decl_visitor >
    {
        using gap::core::crtp< derived, unsup_decl_visitor >::underlying;

        operation visit(const clang_decl *decl, scope_context &scope) {
            return underlying().builder().template compose< unsup::UnsupportedDecl >()
                .bind(underlying().head().location(decl))
                .bind_always(decl_name(decl))
                .freeze();
        }
    };


    template< typename derived >
    struct unsup_stmt_visitor : gap::core::crtp< derived, unsup_stmt_visitor >
    {
        using gap::core::crtp< derived, unsup_stmt_visitor >::underlying;

        operation visit(const clang_stmt *stmt, scope_context &scope) {
            return underlying().builder().template compose< unsup::UnsupportedStmt >()
                .bind(underlying().location(stmt))
                .bind_always(stmt->getStmtClassName())
                .bind_always(return_type(stmt, scope))
                .bind_always(make_children(stmt, scope))
                .freeze();
        }

      private:

        std::vector< builder_callback > make_children(const clang_stmt *stmt, scope_context &scope) {
            std::vector< builder_callback > children;
            for (auto ch : stmt->children()) {
                // For each subexpression, the unsupported operation holds a region.
                // Last value of the region is an operand of the expression.
                children.push_back([this, ch, &scope](auto &bld, auto loc) {
                    underlying().head().visit(ch, scope);
                });
            }
            return children;
        }

        mlir_type return_type(const clang_stmt *stmt, scope_context &scope) {
            auto expr = mlir::dyn_cast_or_null< clang_expr >(stmt);
            return expr ? underlying().head().visit(expr->getType(), scope) : mlir_type();
        }
    };

    template< typename derived >
    struct unsup_type_visitor : gap::core::crtp< derived, unsup_type_visitor >
    {
        using gap::core::crtp< derived, unsup_type_visitor >::underlying;

        mlir_type visit(const clang_type *type, scope_context &scope) {
            return unsup::UnsupportedType::get(&underlying().mcontext(), type->getTypeClassName());
        }

        mlir_type visit(clang_qual_type type, scope_context &scope) {
            VAST_ASSERT(!type.isNull());
            return visit(type.getTypePtr(), scope);
        }
    };


    template< typename derived >
    struct unsup_attr_visitor : gap::core::crtp< derived, unsup_attr_visitor >
    {
        using gap::core::crtp< derived, unsup_attr_visitor >::underlying;

        std::optional< named_attr > visit(const clang_attr *attr, scope_context &scope) {
            std::string spelling(attr->getSpelling());
            auto mctx = &underlying().mcontext();
            return std::make_optional< named_attr >(
                mlir::StringAttr::get(mctx, spelling), unsup::UnsupportedAttr::get(mctx, spelling)
            );
        }
    };

    //
    // composed unsupported visitor
    //
    struct unsup_visitor
        : visitor_base
        , unsup_decl_visitor< unsup_visitor >
        , unsup_stmt_visitor< unsup_visitor >
        , unsup_type_visitor< unsup_visitor >
        , unsup_attr_visitor< unsup_visitor >
    {
        unsup_visitor(visitor_base &head, mcontext_t &mctx, codegen_builder &bld)
            : mctx(mctx), bld(bld), visitors_head(head)
        {}

        operation visit(const clang_decl *decl, scope_context &scope) override {
            return unsup_decl_visitor::visit(decl, scope);
        }

        operation visit(const clang_stmt *stmt, scope_context &scope) override {
            return unsup_stmt_visitor::visit(stmt, scope);
        }

        mlir_type visit(const clang_type *type, scope_context &scope) override {
            return unsup_type_visitor::visit(type, scope);
        }

        mlir_type visit(clang_qual_type type, scope_context &scope) override {
            return unsup_type_visitor::visit(type, scope);
        }

        std::optional< named_attr > visit(const clang_attr *attr, scope_context &scope) override {
            return unsup_attr_visitor::visit(attr, scope);
        }

        operation visit_prototype(const clang_function *decl, scope_context &scope) override {
            return unsup_decl_visitor::visit(decl, scope);
        }

        std::optional< loc_t > location(const clang_decl *) override {
            return mlir::UnknownLoc::get(&mctx);
        }

        std::optional< loc_t > location(const clang_stmt *) override {
            return mlir::UnknownLoc::get(&mctx);
        }

        std::optional< loc_t > location(const clang_expr *) override {
            return mlir::UnknownLoc::get(&mctx);
        }

        std::optional< symbol_name > symbol(clang_global decl) override {
            return decl_name(decl.getDecl());
        }

        std::optional< symbol_name > symbol(const clang_decl_ref_expr *expr) override {
            return decl_name(expr->getDecl());
        }

        mcontext_t& mcontext() { return mctx; }
        codegen_builder& builder() { return bld; }

        visitor_view head() { return visitors_head; }

      protected:
        mcontext_t &mctx;
        codegen_builder &bld;
        visitor_view visitors_head;
    };

} // namespace vast::cg
