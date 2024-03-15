// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/CodeGen/CodeGenVisitorBase.hpp"
#include "vast/CodeGen/FallBackVisitor.hpp"

namespace vast::cg
{
    //
    // default codegen visitor configuration
    //
    struct codegen_visitor final : fallback_visitor
    {
        using fallback_visitor::fallback_visitor;
        using fallback_visitor::visit;
    };

} // namespace vast::cg
