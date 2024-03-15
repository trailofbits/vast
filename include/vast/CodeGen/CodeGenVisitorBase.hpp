// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <clang/AST/AttrVisitor.h>
#include <clang/AST/DeclVisitor.h>
#include <clang/AST/StmtVisitor.h>
#include <clang/AST/TypeVisitor.h>
VAST_UNRELAX_WARNINGS

#include "vast/Util/Common.hpp"
#include "vast/CodeGen/CodeGenMeta.hpp"
#include "vast/CodeGen/CodeGenBuilder.hpp"

namespace vast::cg {

    using clang_decl = clang::Decl;
    using clang_stmt = clang::Stmt;
    using clang_expr = clang::Expr;
    using clang_type = clang::Type;
    using clang_attr = clang::Attr;

    using clang_function_type = clang::FunctionType;
    using clang_function_proto_type = clang::FunctionProtoType;

    using clang_qual_type = clang::QualType;

    template< typename derived_t >
    using decl_visitor_base = clang::ConstDeclVisitor< derived_t, operation >;

    template< typename derived_t >
    using stmt_visitor_base = clang::ConstStmtVisitor< derived_t, operation >;

    template< typename derived_t >
    using type_visitor_base = clang::TypeVisitor< derived_t, mlir_type >;

    template< typename derived_t >
    using attr_visitor_base = clang::ConstAttrVisitor< derived_t, mlir_attr >;

    //
    // Classes derived from `visitor_base` are used to visit clang AST nodes
    //
    struct visitor_base
    {
        visitor_base(mcontext_t &mctx, meta_generator &meta)
            : mctx(mctx), meta(meta), bld(&mctx)
        {}

        virtual ~visitor_base() = default;

        virtual operation visit(const clang_decl *) = 0;
        virtual operation visit(const clang_stmt *) = 0;
        virtual mlir_type visit(const clang_type *) = 0;
        virtual mlir_type visit(clang_qual_type)    = 0;
        virtual mlir_attr visit(const clang_attr *) = 0;

        virtual mlir_type visit(const clang_function_type *, bool /* is_variadic */);
        virtual mlir_type visit_as_lvalue_type(clang_qual_type);

        mcontext_t& mcontext() { return mctx; }
        const mcontext_t& mcontext() const { return mctx; }

        loc_t location(const auto *node) const { return meta.location(node); }

        codegen_builder& builder() { return bld; }

      protected:
        mcontext_t &mctx;
        meta_generator &meta;

        // TODO figure out how to make scoped visitor that initilizes builder to
        // specific scopes
        codegen_builder bld;
    };

    using visitor_base_ptr = std::unique_ptr< visitor_base >;

} // namespace vast::cg
