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
        operation visit(const clang_decl *decl);
    };


    struct unsup_stmt_visitor : unsup_visitor_base
    {
        using unsup_visitor_base::unsup_visitor_base;
        operation visit(const clang_stmt *stmt);

      private:
        std::vector< BuilderCallBackFn > make_children(const clang_stmt *stmt);
        mlir_type return_type(const clang_stmt *stmt);
    };


    struct unsup_type_visitor : unsup_visitor_base
    {
        using unsup_visitor_base::unsup_visitor_base;

        mlir_type visit(const clang_type *type);
        mlir_type visit(clang_qual_type type) {
            VAST_ASSERT(!type.isNull());
            return visit(type.getTypePtr());
        }
    };


    struct unsup_attr_visitor : unsup_visitor_base
    {
        using unsup_visitor_base::unsup_visitor_base;
        mlir_attr visit(const clang_attr *attr);
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
        unsup_visitor(mcontext_t &mctx, codegen_builder &bld, meta_generator &mg, symbol_generator &sg)
            : visitor_base(mctx, mg, sg)
            , unsup_decl_visitor(bld, visitor_view(*this))
            , unsup_stmt_visitor(bld, visitor_view(*this))
            , unsup_type_visitor(bld, visitor_view(*this))
            , unsup_attr_visitor(bld, visitor_view(*this))
        {}

        operation visit(const clang_decl *decl) override {
            return unsup_decl_visitor::visit(decl);
        }

        operation visit(const clang_stmt *stmt) override {
            return unsup_stmt_visitor::visit(stmt);
        }

        mlir_type visit(const clang_type *type) override {
            return unsup_type_visitor::visit(type);
        }

        mlir_type visit(clang_qual_type type) override {
            return unsup_type_visitor::visit(type);
        }

        mlir_attr visit(const clang_attr *attr) override {
            return unsup_attr_visitor::visit(attr);
        }

        operation visit_prototype(const clang_function *decl) override {
            return unsup_decl_visitor::visit(decl);
        }
    };

} // namespace vast::cg
