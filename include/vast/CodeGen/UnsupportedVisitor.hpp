// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

#include "vast/CodeGen/CodeGenVisitorBase.hpp"

namespace vast::cg
{
    struct unsup_visitor_base  {
        unsup_visitor_base(codegen_builder &bld, visitor_view self)
            : bld(bld), self(self)
        {}

      protected:
        codegen_builder &bld;
        visitor_view self;
    };

    struct unsup_decl_visitor : unsup_visitor_base
    {
        using unsup_visitor_base::unsup_visitor_base;
        operation visit(const clang_decl *decl, scope_context &scope);
    };


    struct unsup_stmt_visitor : unsup_visitor_base
    {
        using unsup_visitor_base::unsup_visitor_base;
        operation visit(const clang_stmt *stmt, scope_context &scope);

      private:
        std::vector< builder_callback > make_children(const clang_stmt *stmt, scope_context &scope);
        mlir_type return_type(const clang_stmt *stmt, scope_context &scope);
    };


    struct unsup_type_visitor : unsup_visitor_base
    {
        using unsup_visitor_base::unsup_visitor_base;

        mlir_type visit(const clang_type *type, scope_context &scope);
        mlir_type visit(clang_qual_type type, scope_context &scope);
    };


    struct unsup_attr_visitor : unsup_visitor_base
    {
        using unsup_visitor_base::unsup_visitor_base;
        mlir_attr visit(const clang_attr *attr, scope_context &scope);
    };


    //
    // composed unsupported visitor
    //
    struct unsup_visitor final
        : visitor_base
        , unsup_decl_visitor
        , unsup_stmt_visitor
        , unsup_type_visitor
        , unsup_attr_visitor
    {
        unsup_visitor(mcontext_t &mctx, codegen_builder &bld, visitor_view self)
            : visitor_base(mctx, self.options())
            , unsup_decl_visitor(bld, self)
            , unsup_stmt_visitor(bld, self)
            , unsup_type_visitor(bld, self)
            , unsup_attr_visitor(bld, self)
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

        mlir_attr visit(const clang_attr *attr, scope_context &scope) override {
            return unsup_attr_visitor::visit(attr, scope);
        }

        operation visit_prototype(const clang_function *decl, scope_context &scope) override {
            return unsup_decl_visitor::visit(decl, scope);
        }

    };

} // namespace vast::cg
