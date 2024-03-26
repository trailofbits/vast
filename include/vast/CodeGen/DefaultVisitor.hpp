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
        , cached_default_type_visitor
        , default_attr_visitor
    {
        default_visitor(mcontext_t &mctx, codegen_builder &bld, meta_generator &mg, symbol_generator &sg, visitor_view self)
            : visitor_base(mctx, mg, sg)
            , default_decl_visitor(bld, self)
            , default_stmt_visitor(bld, self)
            , cached_default_type_visitor(bld, self)
            , default_attr_visitor(bld, self)
        {}

        operation visit(const clang_decl *decl) override {
            return default_decl_visitor::visit(decl);
        }

        operation visit(const clang_stmt *stmt) override {
            return default_stmt_visitor::visit(stmt);
        }

        mlir_type visit(const clang_type *type) override {
            return cached_default_type_visitor::visit(type);
        }

        mlir_type visit(clang_qual_type type) override {
            return cached_default_type_visitor::visit(type);
        }

        mlir_attr visit(const clang_attr *attr) override {
            return default_attr_visitor::visit(attr);
        }

        operation visit_prototype(const clang_function *decl) override {
            return default_decl_visitor::visit_prototype(decl);
        }
    };

} // namespace vast::cg
