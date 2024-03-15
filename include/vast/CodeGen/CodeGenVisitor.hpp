// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/CodeGen/CodeGenScope.hpp"
#include "vast/Util/Warnings.hpp"

#include "vast/CodeGen/CodeGenVisitorBase.hpp"
#include "vast/CodeGen/FallBackVisitor.hpp"

namespace vast::cg
{
    //
    // Classes derived from `codegen_visitor_base` extend the `visitor_base` with codegen
    // specific methods. Codegen visitor is a fallback visitor, which means that it can be
    // initialized with a chain of other visitors, which will be used as a fallback in case
    // that the first `visit` method is unsuccessful.
    //
    struct codegen_visitor_base : fallback_visitor
    {
        using fallback_visitor::fallback_visitor;
        virtual ~codegen_visitor_base() = default;

        using visitor_base::visit;

        virtual mlir_type visit(const clang_function_type *, bool /* is_variadic */) = 0;
        virtual mlir_type visit_as_lvalue_type(clang_qual_type) = 0;
    };

    //
    // default codegen visitor configuration
    //
    struct codegen_visitor : codegen_visitor_base
    {
        using codegen_visitor_base::codegen_visitor_base;

        using codegen_visitor_base::visit;

        mlir_type visit(const clang_function_type *fty, bool is_variadic) override;
        mlir_type visit_as_lvalue_type(clang_qual_type ty) override;
    };

} // namespace vast::cg
