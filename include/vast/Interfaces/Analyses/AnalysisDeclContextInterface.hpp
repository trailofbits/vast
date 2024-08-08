// Copyright (c) 2024-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/OperationSupport.h>
VAST_RELAX_WARNINGS

#include "vast/Interfaces/AST/ASTContextInterface.hpp"
#include "vast/Interfaces/AST/DeclInterface.hpp"
#include "vast/Interfaces/CFG/CFGInterface.hpp"

/// Include the generated interface declarations.
#include "vast/Interfaces/Analyses/AnalysisDeclContextInterface.h.inc"

namespace vast::analyses {

    template< typename T >
    T *AnalysisDeclContextInterface::getAnalysis() {
        return nullptr;
    }

} // namespace vast::analyses
