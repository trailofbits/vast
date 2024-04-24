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

        operation visit(const clang_decl *decl);
        operation visit(const clang_stmt *stmt);
        mlir_type visit(const clang_type *type);
        mlir_type visit(clang_qual_type ty);
        mlir_attr visit(const clang_attr *attr);

        operation visit_prototype(const clang_function *decl);

        mlir_type visit(const clang_function_type *fty, bool is_variadic);

        mlir_type visit_as_lvalue_type(clang_qual_type ty);

        loc_t location(const auto *node) const;

        std::optional< symbol_name > symbol(auto &&decl);

        mcontext_t& mcontext();
        const mcontext_t& mcontext() const;

        visitor_base *raw() { return &visitor; }

      protected:
        visitor_base &visitor;
    };


    struct scoped_visitor_view : visitor_view
    {
        scoped_visitor_view(visitor_base &visitor, scope_context &scope)
            : visitor_view(visitor), scope(scope)
        {}

        scope_context &scope;
    };


    struct clang_visitor_base
    {
        clang_visitor_base(codegen_builder &bld, scoped_visitor_view self)
            : bld(bld), self(self)
        {}

        template< typename Builder >
        auto declare(Builder &&bld) -> decltype(bld()) {
            return self.scope.declare(std::forward< Builder >(bld));
        }

      protected:
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
        visitor_base(mcontext_t &mctx, meta_generator &mg, symbol_generator &sg)
            : mctx(mctx), mg(mg), sg(sg)
        {}

        virtual ~visitor_base() = default;

        virtual operation visit(const clang_decl *) = 0;
        virtual operation visit(const clang_stmt *) = 0;
        virtual mlir_type visit(const clang_type *) = 0;
        virtual mlir_type visit(clang_qual_type)    = 0;
        virtual mlir_attr visit(const clang_attr *) = 0;

        virtual mlir_type visit(const clang_function_type *, bool /* is_variadic */);
        virtual mlir_type visit_as_lvalue_type(clang_qual_type);

        virtual operation visit_prototype(const clang_function *decl) = 0;

        mcontext_t& mcontext() { return mctx; }
        const mcontext_t& mcontext() const { return mctx; }

        loc_t location(const auto *node) const { return mg.location(node); }

        std::optional< symbol_name > symbol(auto &&decl) {
            return sg.symbol(std::forward< decltype(decl) >(decl));
        }

      protected:
        mcontext_t &mctx;
        meta_generator &mg;
        symbol_generator &sg;
    };

    using visitor_base_ptr = std::unique_ptr< visitor_base >;

    loc_t visitor_view::location(const auto *node) const {
        return visitor.location(node);
    }

    std::optional< symbol_name > visitor_view::symbol(auto &&decl) {
        return visitor.symbol(std::forward< decltype(decl) >(decl));
    }

} // namespace vast::cg
