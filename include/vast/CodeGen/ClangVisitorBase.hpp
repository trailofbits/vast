// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <clang/AST/AttrVisitor.h>
#include <clang/AST/DeclVisitor.h>
#include <clang/AST/StmtVisitor.h>
#include <clang/AST/TypeVisitor.h>
VAST_UNRELAX_WARNINGS

#include "vast/CodeGen/CodeGenVisitorBase.hpp"
#include "vast/CodeGen/CodeGenBuilder.hpp"

namespace vast::cg {

    struct clang_visitor_base
    {
        clang_visitor_base(mcontext_t &mctx, codegen_builder &bld, visitor_view self, scope_context &scope)
            : mctx(mctx), bld(bld), self(self, scope)
        {}

        template< typename builder_t >
        auto maybe_declare(const clang_named_decl *decl, builder_t &&bld) -> decltype(bld()) {
            return self.scope.maybe_declare(decl, std::forward< builder_t >(bld));
        }

        bool is_declared_type(const clang_named_decl *decl) const {
            return self.scope.is_declared_type(decl);
        }

        // expects a range of values (operations returning a single value)
        template< typename range_type >
        values_t visit_values_range(range_type &&range) {
            values_t values;
            for (auto item : range) {
                values.push_back(self.visit(item)->getResult(0));
            }
            return values;
        }

        template< typename yield_type >
        auto mk_stmt_builder(const clang_stmt *stmt) {
            return [this, stmt] (auto &state, auto loc) {
                self.visit(stmt);
                auto &op = state.getBlock()->back();
                VAST_ASSERT(op.getNumResults() == 1);
                bld.create< yield_type >(loc, op.getResult(0));
            };
        }

        template< typename yield_type >
        auto mk_stmt_builder_with_args(const clang_stmt *stmt, const mlir_type &types...) {
            return [&, this, stmt] (auto &state, auto loc) {
                state.getBlock()->getParent()->addArgument(types, loc);
                self.visit(stmt);
                auto &op = state.getBlock()->back();
                VAST_ASSERT(op.getNumResults() == 1);
                bld.create< yield_type >(loc, op.getResult(0));
            };
        }

        auto mk_value_builder(const clang_stmt *stmt) {
            return mk_stmt_builder< hl::ValueYieldOp >(stmt);
        }

        auto mk_value_builder_with_args(const clang_stmt *stmt, const mlir_type &types...) {
            return mk_stmt_builder_with_args< hl::ValueYieldOp >(stmt, types);
        }

        auto mk_cond_builder(const clang_stmt *stmt) {
            return mk_stmt_builder< hl::CondYieldOp >(stmt);
        }

        auto mk_cond_builder_with_args(const clang_stmt *stmt, const mlir_type &types...) {
            return mk_stmt_builder_with_args< hl::CondYieldOp >(stmt, types);
        }

        auto mk_true_yielder() {
            return [this] (auto &, auto loc) {
               bld.create< hl::CondYieldOp >(loc, bld.true_value(loc));
            };
        }

        auto mk_false_yielder() {
            return [this] (auto &, auto loc) {
                bld.create< hl::CondYieldOp >(loc, bld.false_value(loc));
            };
        }

        auto mk_region_builder(const clang_stmt *stmt) {
            return [this, stmt] (auto &bld, auto) {
                self.visit(stmt);
            };
        }

        auto mk_optional_region_builder(const clang_stmt *stmt) {
            return [this, stmt] (auto &bld, auto) {
                if (stmt) self.visit(stmt);
            };
        }

        auto mk_decl_context_builder(const clang_decl_context *ctx) {
            return [this, ctx] (auto &, auto) {
                for (auto decl : ctx->decls()) {
                    self.visit(decl);
                }
            };
        }

        auto mk_type_yield_builder(const clang_expr *expr) {
            return mk_stmt_builder< hl::TypeYieldOp >(expr);
        }

      protected:
        mcontext_t &mctx;
        codegen_builder &bld;
        scoped_visitor_view self;
    };

    template< typename derived_t >
    struct decl_visitor_base : clang_visitor_base, clang::ConstDeclVisitor< derived_t, operation > {
        using clang_visitor_base::clang_visitor_base;
    };

    template< typename derived_t >
    struct stmt_visitor_base : clang_visitor_base, clang::ConstStmtVisitor< derived_t, operation > {
        using clang_visitor_base::clang_visitor_base;
    };

    template< typename derived_t >
    struct type_visitor_base : clang_visitor_base, clang::TypeVisitor< derived_t, mlir_type > {
        using clang_visitor_base::clang_visitor_base;
    };

    template< typename derived_t >
    struct attr_visitor_base : clang_visitor_base, clang::ConstAttrVisitor< derived_t, mlir_attr > {
        using clang_visitor_base::clang_visitor_base;
    };

} // namespace vast::cg
