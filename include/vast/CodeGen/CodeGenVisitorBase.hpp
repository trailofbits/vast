// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <clang/AST/AttrVisitor.h>
#include <clang/AST/DeclVisitor.h>
#include <clang/AST/StmtVisitor.h>
#include <clang/AST/TypeVisitor.h>
VAST_UNRELAX_WARNINGS

#include "vast/CodeGen/CodeGenBuilder.hpp"
#include "vast/CodeGen/CodeGenMeta.hpp"
#include "vast/CodeGen/Common.hpp"
#include "vast/CodeGen/ScopeContext.hpp"
#include "vast/CodeGen/SymbolGenerator.hpp"

namespace vast::cg {

    struct visitor_base;

    struct visitor_view
    {
        explicit visitor_view(visitor_base &visitor) : visitor(visitor) {}

        operation visit(const clang_decl *decl, scope_context &scope);
        operation visit(const clang_stmt *stmt, scope_context &scope);
        mlir_type visit(const clang_type *type, scope_context &scope);
        mlir_type visit(clang_qual_type ty, scope_context &scope);
        mlir_attr visit(const clang_attr *attr, scope_context &scope);

        operation visit_prototype(const clang_function *decl, scope_context &scope);

        std::optional< loc_t > location(const auto *node) const;
        loc_t maybe_location(const auto *node);

        std::optional< symbol_name > symbol(auto &&decl);

        mcontext_t& mcontext();
        const mcontext_t& mcontext() const;

        visitor_base *raw() { return &visitor; }

      protected:
        visitor_base &visitor;
    };


    struct scoped_visitor_view : visitor_view
    {
        explicit scoped_visitor_view(visitor_base &visitor, scope_context &scope)
            : visitor_view(visitor), scope(scope)
        {}

        explicit scoped_visitor_view(visitor_view view, scope_context &scope)
            : visitor_view(std::move(view)), scope(scope)
        {}

        operation visit(const clang_decl *decl);
        operation visit(const clang_stmt *stmt);
        mlir_type visit(const clang_type *type);
        mlir_type visit(clang_qual_type ty);
        mlir_attr visit(const clang_attr *attr);

        operation visit_prototype(const clang_function *decl);

        scope_context &scope;
    };

    struct clang_visitor_base
    {
        clang_visitor_base(mcontext_t &mctx, codegen_builder &bld, visitor_view self, scope_context &scope)
            : mctx(mctx), bld(bld), self(self, scope)
        {}

        template< typename builder_t >
        auto maybe_declare(builder_t &&bld) -> decltype(bld()) {
            return self.scope.maybe_declare(std::forward< builder_t >(bld));
        }

        bool is_declared_type(string_ref name) const {
            return self.scope.is_declared_type(name);
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

        auto mk_value_builder(const clang_stmt *stmt) {
            return mk_stmt_builder< hl::ValueYieldOp >(stmt);
        }

        auto mk_cond_builder(const clang_stmt *stmt) {
            return mk_stmt_builder< hl::CondYieldOp >(stmt);
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

    //
    // Classes derived from `visitor_base` are used to visit clang AST nodes
    //
    struct visitor_base
    {
        visitor_base(mcontext_t &mctx) : mctx(mctx) {}

        virtual ~visitor_base() = default;

        virtual operation visit(const clang_decl *, scope_context &scope) = 0;
        virtual operation visit(const clang_stmt *, scope_context &scope) = 0;
        virtual mlir_type visit(const clang_type *, scope_context &scope) = 0;
        virtual mlir_type visit(clang_qual_type, scope_context &scope)    = 0;
        virtual mlir_attr visit(const clang_attr *, scope_context &scope) = 0;

        // virtual mlir_type visit_as_lvalue_type(clang_qual_type, scope_context &scope);

        virtual operation visit_prototype(const clang_function *decl, scope_context &scope) = 0;

        mcontext_t& mcontext() { return mctx; }
        const mcontext_t& mcontext() const { return mctx; }

        virtual std::optional< loc_t > location(const clang_decl *) { return std::nullopt; }
        virtual std::optional< loc_t > location(const clang_stmt *) { return std::nullopt; }
        virtual std::optional< loc_t > location(const clang_expr *) { return std::nullopt; }

        virtual std::optional< symbol_name > symbol(clang_global decl) { return std::nullopt; }
        virtual std::optional< symbol_name > symbol(const clang_decl_ref_expr *decl) { return std::nullopt; }

      protected:
        mcontext_t &mctx;
    };

    using visitor_base_ptr = std::unique_ptr< visitor_base >;

    std::optional< loc_t > visitor_view::location(const auto *node) const {
        return visitor.location(node);
    }

    loc_t visitor_view::maybe_location(const auto *node) {
        if (auto loc = visitor.location(node))
            return loc.value();
        return mlir::UnknownLoc::get(&mcontext());
    }

    std::optional< symbol_name > visitor_view::symbol(auto &&decl) {
        return visitor.symbol(std::forward< decltype(decl) >(decl));
    }

} // namespace vast::cg
