// Copyright (c) 2024-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <clang/AST/Attr.h>
#include <clang/AST/Decl.h>
#include <clang/AST/Stmt.h>
#include <clang/AST/Type.h>
VAST_UNRELAX_WARNINGS

#include "vast/CodeGen/CodeGenVisitorBase.hpp"

namespace vast::cg {

    struct visitor_view
    {
        explicit visitor_view(visitor_base &visitor) : visitor(visitor) {}

        void Visit(clang::Decl *decl) { visitor.Visit(decl); }
        void Visit(clang::Stmt *stmt) { visitor.Visit(stmt); }
        void Visit(clang::Type *type) { visitor.Visit(type); }
        void Visit(clang::Attr *attr) { visitor.Visit(attr); }

      private:
        visitor_base &visitor;
    };

} // namespace vast::cg
