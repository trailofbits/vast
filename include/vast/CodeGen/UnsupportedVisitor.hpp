// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

#include "vast/CodeGen/CodeGenVisitorBase.hpp"
#include "vast/CodeGen/VisitorView.hpp"

namespace vast::cg
{
    struct unsup_decl_visitor
    {
        unsup_decl_visitor(base_visitor_view self) : self(self) {}

        operation visit(const clang_decl *decl) {
            VAST_UNIMPLEMENTED;
        }

      private:
        base_visitor_view self;
    };


    struct unsup_stmt_visitor
    {
        unsup_stmt_visitor(base_visitor_view self) : self(self) {}

        operation visit(const clang_stmt *stmt) {
            VAST_UNIMPLEMENTED;
        }

      private:
        base_visitor_view self;
    };


    struct unsup_type_visitor
    {
        unsup_type_visitor(base_visitor_view self) : self(self) {}

        mlir_type visit(const clang_type *type) { return make_type(type); }

        mlir_type visit(clang_qual_type type) {
            VAST_ASSERT(!type.isNull());
            return visit(type.getTypePtr());
        }

      private:
        mlir_type make_type(const clang_type *type);

        base_visitor_view self;
    };


    struct unsup_attr_visitor
    {
        unsup_attr_visitor(base_visitor_view self) : self(self) {}

        mlir_attr visit(const clang_attr *attr) {
            VAST_UNIMPLEMENTED;
        }

      private:
        base_visitor_view self;
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
        unsup_visitor(mcontext_t &mctx, meta_generator &meta)
            : visitor_base(mctx, meta)
            , unsup_decl_visitor(base_visitor_view(*this))
            , unsup_stmt_visitor(base_visitor_view(*this))
            , unsup_type_visitor(base_visitor_view(*this))
            , unsup_attr_visitor(base_visitor_view(*this))
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
    };

} // namespace vast::cg
