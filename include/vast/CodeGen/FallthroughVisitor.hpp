// Copyright (c) 2024, Trail of Bits, Inc.

#pragma once

#include "vast/CodeGen/CodeGenVisitorBase.hpp"

namespace vast::cg
{

  struct fallthrough_visitor : visitor_base {

    using visitor_base::visitor_base;

    operation visit(const clang_decl *, scope_context &) override { return {}; }
    operation visit(const clang_stmt *, scope_context &) override { return {}; }
    mlir_type visit(const clang_type *, scope_context &) override { return {}; }
    mlir_type visit(clang_qual_type, scope_context &)    override { return {}; }
    std::optional< named_attr > visit(const clang_attr *, scope_context &) override { return {}; }

    operation visit_prototype(const clang_function *, scope_context &) override { return {}; }

    std::optional< loc_t > location(const clang_decl *) override { return {}; }
    std::optional< loc_t > location(const clang_stmt *) override { return {}; }
    std::optional< loc_t > location(const clang_expr *) override { return {}; }

    std::optional< symbol_name > symbol(clang_global) override { return {}; }
    std::optional< symbol_name > symbol(const clang_decl_ref_expr *) override { return {}; }
};

} // namespace vast::cg