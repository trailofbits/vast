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

namespace vast::cg {

    using clang_decl = clang::Decl;
    using clang_stmt = clang::Stmt;
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

    struct visitor_base
    {
        virtual ~visitor_base() = default;

        virtual operation visit(clang_decl *decl) = 0;
        virtual operation visit(clang_stmt *stmt) = 0;
        virtual mlir_type visit(clang_type *type) = 0;
        virtual mlir_type visit(clang_qual_type attr) = 0;
        virtual mlir_attr visit(clang_attr *attr) = 0;

        virtual mlir_type visit_function_type(const clang_function_type *fty, bool is_variadic) = 0;
        virtual mlir_type visit_as_lvalue_type(clang_qual_type ty) = 0;
    };

} // namespace vast::cg
