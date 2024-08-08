// Copyright (c) 2024-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/OperationSupport.h>
VAST_RELAX_WARNINGS

#include "vast/Analyses/Iterators.hpp"
#include "vast/Interfaces/AST/StmtInterface.hpp"
#include <optional>

/// Include the generated interface declarations.
#include "vast/Interfaces/CFG/CFGInterface.h.inc"

namespace vast::cfg {

    template< typename Callback >
    void CFGInterface::VisitBlockStmts(Callback &) {

    }

    template< typename T >
    std::optional< T > CFGElementInterface::getAs() {
        return {};
    }

} // namespace vast::cfg
