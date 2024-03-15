// Copyright (c) 2023-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <clang/AST/Attr.h>
VAST_UNRELAX_WARNINGS

#include "vast/CodeGen/CodeGenVisitorBase.hpp"
#include "vast/CodeGen/VisitorView.hpp"

namespace vast::cg {

    struct default_attr_visitor
    {
        explicit default_attr_visitor(visitor_view self) : self(self) {}

        mlir_attr visit(const clang_attr *decl) { return {}; }

      private:
        visitor_view self;
    };

} // namespace vast::cg