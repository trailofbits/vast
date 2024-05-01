// Copyright (c) 2024-present, Trail of Bits, Inc.

#pragma once

#include "vast/CodeGen/CodeGenVisitorBase.hpp"
#include "vast/CodeGen/DefaultAttrVisitor.hpp"
#include "vast/CodeGen/DefaultDeclVisitor.hpp"
#include "vast/CodeGen/DefaultStmtVisitor.hpp"
#include "vast/CodeGen/DefaultTypeVisitor.hpp"

namespace vast::cg
{
    struct default_visitor final : visitor_base
    {
        default_visitor(mcontext_t &mctx, codegen_builder &bld, meta_generator &mg, symbol_generator &sg, visitor_view self)
            : visitor_base(mctx, mg, sg), bld(bld), self(self)
        {}

        operation visit(const clang_decl *decl, scope_context &scope) override {
            default_decl_visitor visitor(bld, self, scope);
            return visitor.visit(decl);
        }

        operation visit(const clang_stmt *stmt, scope_context &scope) override {
            default_stmt_visitor visitor(bld, self, scope);
            return visitor.visit(stmt);
        }

        mlir_type visit(const clang_type *type, scope_context &scope) override {
            if (auto value = cache.lookup(type)) {
                return value;
            }

            default_type_visitor visitor(bld, self, scope);
            auto result = visitor.visit(type);
            cache.try_emplace(type, result);
            return result;
        }

        mlir_type visit(clang_qual_type type, scope_context &scope) override {
            if (auto value = qual_cache.lookup(type)) {
                return value;
            }

            default_type_visitor visitor(bld, self, scope);
            auto result = visitor.visit(type);
            qual_cache.try_emplace(type, result);
            return result;
        }

        mlir_attr visit(const clang_attr *attr, scope_context &scope) override {
            default_attr_visitor visitor(bld, self, scope);
            return visitor.visit(attr);
        }

        operation visit_prototype(const clang_function *decl, scope_context &scope) override {
            default_decl_visitor visitor(bld, self, scope);
            return visitor.visit_prototype(decl);
        }

        llvm::DenseMap< const clang_type *, mlir_type > cache;
        llvm::DenseMap< clang_qual_type, mlir_type > qual_cache;

        codegen_builder &bld;
        visitor_view self;
    };

} // namespace vast::cg
