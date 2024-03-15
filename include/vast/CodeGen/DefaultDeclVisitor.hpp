// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <clang/AST/DeclVisitor.h>
VAST_UNRELAX_WARNINGS

#include "vast/CodeGen/CodeGenVisitorBase.hpp"
#include "vast/CodeGen/VisitorView.hpp"

namespace vast::cg {

    struct default_decl_visitor
    {
        explicit default_decl_visitor(base_visitor_view self) : self(self) {}

        operation visit(const clang_decl *decl) { return {}; }

      private:
        base_visitor_view self;
    };

} // namespace vast::cg