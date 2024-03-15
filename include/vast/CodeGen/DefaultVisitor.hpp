// Copyright (c) 2024-present, Trail of Bits, Inc.

#pragma once

#include "vast/CodeGen/CodeGenVisitorBase.hpp"
#include "vast/CodeGen/DefaultAttrVisitor.hpp"
#include "vast/CodeGen/DefaultDeclVisitor.hpp"
#include "vast/CodeGen/DefaultStmtVisitor.hpp"
#include "vast/CodeGen/DefaultTypeVisitor.hpp"

namespace vast::cg
{
    struct default_visitor final
        : visitor_base
        , default_decl_visitor
        , default_stmt_visitor
        , default_type_visitor_with_dl
        , default_attr_visitor
    {
        default_visitor(mcontext_t &mctx, meta_generator &meta)
            : visitor_base(mctx, meta)
            , default_decl_visitor(base_visitor_view(*this))
            , default_stmt_visitor(base_visitor_view(*this))
            , default_type_visitor_with_dl(base_visitor_view(*this))
            , default_attr_visitor(base_visitor_view(*this))
        {}

        operation visit(const clang_decl *decl) override {
            return default_decl_visitor::visit(decl);
        }

        operation visit(const clang_stmt *stmt) override {
            return default_stmt_visitor::visit(stmt);
        }

        mlir_type visit(const clang_type *type) override {
            return default_type_visitor_with_dl::visit(type);
        }

        mlir_type visit(clang_qual_type type) override {
            return default_type_visitor_with_dl::visit(type);
        }

        mlir_attr visit(const clang_attr *attr) override {
            return default_attr_visitor::visit(attr);
        }
    };

} // namespace vast::cg
