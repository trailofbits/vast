// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Common.hpp"
#include "vast/CodeGen/CodeGenMeta.hpp"
#include "vast/CodeGen/CodeGenBuilder.hpp"
#include "vast/CodeGen/CodeGenVisitorBase.hpp"

#include "vast/Dialect/HighLevel/HighLevelAttributes.hpp"

#include "vast/CodeGen/VisitorView.hpp"

namespace vast::cg {

    struct default_type_visitor
    {
        explicit default_type_visitor(base_visitor_view self) : self(self) {}

        mlir_type visit(const clang_type *type) { return {}; }
        mlir_type visit(clang_qual_type type) { return {}; }

      private:
        base_visitor_view self;
    };

    struct default_type_visitor_with_dl : default_type_visitor
    {
        using default_type_visitor::default_type_visitor;

        mlir_type visit(const clang_type *type) {
            return default_type_visitor::visit(type);
        }

        mlir_type visit(clang_qual_type type) {
            return default_type_visitor::visit(type);
        }
    };

} // namespace vast::cg
