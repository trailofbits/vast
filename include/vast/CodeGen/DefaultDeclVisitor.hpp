// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <clang/AST/DeclVisitor.h>
VAST_UNRELAX_WARNINGS

#include "vast/CodeGen/CodeGenVisitorBase.hpp"
#include "vast/CodeGen/VisitorView.hpp"

namespace vast::cg {

    struct default_decl_visitor : decl_visitor_base< default_decl_visitor >
    {
        explicit default_decl_visitor(visitor_view self) : self(self) {}

        using decl_visitor_base< default_decl_visitor >::Visit;

        operation visit(const clang_decl *decl) { return Visit(decl); }
        operation visit_prototype(const clang::FunctionDecl *decl);

        operation VisitFunctionDecl(const clang::FunctionDecl *decl);

      private:
        visitor_view self;
    };

} // namespace vast::cg