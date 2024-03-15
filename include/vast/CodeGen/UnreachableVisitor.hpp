// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

#include "vast/CodeGen/CodeGenVisitorBase.hpp"

namespace vast::cg
{
    //
    // This is a bottom visitor, which yields an error if called
    //
    struct unreach_visitor final : visitor_base
    {
        using visitor_base::visitor_base;

        operation visit(const clang_decl *decl) override {
            VAST_FATAL("unsupported decl: {0}", decl->getDeclKindName());
        }

        operation visit(const clang_stmt *stmt) override {
            VAST_FATAL("unsupported stmt: {0}", stmt->getStmtClassName());
        }

        mlir_type visit(const clang_type *type) override {
            VAST_FATAL("unsupported type: {0}", type->getTypeClassName());
        }

        mlir_type visit(clang_qual_type type) override {
            VAST_FATAL("unsupported type: {0}", type.getAsString());
        }

        mlir_attr visit(const clang_attr *attr) override {
            VAST_FATAL("unsupported attr: {0}", attr->getSpelling());
        }
    };


} // namespace vast::cg
