// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

#include "vast/CodeGen/CodeGenVisitorBase.hpp"

namespace vast::cg
{
    //
    // This is a bottom visitor, which yields an error if called
    //
    struct unreach_visitor : visitor_base
    {
        using visitor_base::visitor_base;

        operation visit(const clang_decl *decl, scope_context &) override {
            VAST_FATAL("unsupported decl: {0}", decl->getDeclKindName());
        }

        operation visit(const clang_stmt *stmt, scope_context &) override {
            VAST_FATAL("unsupported stmt: {0}", stmt->getStmtClassName());
        }

        mlir_type visit(const clang_type *type, scope_context &) override {
            VAST_FATAL("unsupported type: {0}", type->getTypeClassName());
        }

        mlir_type visit(clang_qual_type type, scope_context &) override {
            VAST_FATAL("unsupported type: {0}", type.getAsString());
        }

        std::optional< named_attr > visit(const clang_attr *attr, scope_context &) override {
            VAST_FATAL("unsupported attr: {0}", attr->getSpelling());
        }

        operation visit_prototype(const clang_function *decl, scope_context &) override {
            VAST_FATAL("unsupported prototype: {0}", decl->getName());
        }

        std::optional< loc_t > location(const clang_decl *decl) override {
            VAST_FATAL("unsupported location for: {0}", decl->getDeclKindName());
        }

        std::optional< loc_t > location(const clang_stmt *stmt) override {
            VAST_FATAL("unsupported location for: {0}", stmt->getStmtClassName());
        }

        std::optional< loc_t > location(const clang_expr *expr) override {
            VAST_FATAL("unsupported location for: {0}", expr->getStmtClassName());
        }

        std::optional< symbol_name > symbol(clang_global decl) override {
            VAST_FATAL("unsupported symbol for: {0}", decl.getDecl()->getDeclKindName());
        }

        std::optional< symbol_name > symbol(const clang_decl_ref_expr *expr) override {
            VAST_FATAL("unsupported symbol for: {0}", expr->getStmtClassName());
        }
    };


} // namespace vast::cg
