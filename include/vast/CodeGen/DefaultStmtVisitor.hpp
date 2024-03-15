// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <clang/AST/StmtVisitor.h>
#include <clang/AST/OperationKinds.h>
VAST_UNRELAX_WARNINGS

#include "vast/CodeGen/CodeGenVisitorBase.hpp"
#include "vast/CodeGen/VisitorView.hpp"

namespace vast::cg {

    struct default_stmt_visitor
    {
        explicit default_stmt_visitor(base_visitor_view self) : self(self) {}

        operation visit(const clang_stmt *stmt) { return {}; }

      private:
        base_visitor_view self;
    };

} // namespace vast::cg
