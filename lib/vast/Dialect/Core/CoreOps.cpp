// Copyright (c) 2022-present, Trail of Bits, Inc.

#include "vast/Dialect/Core/CoreDialect.hpp"
#include "vast/Dialect/Core/CoreOps.hpp"
#include "vast/Dialect/Core/CoreTypes.hpp"
#include "vast/Dialect/Core/CoreAttributes.hpp"

#include "vast/Util/Common.hpp"
#include "vast/Util/Dialect.hpp"
#include "vast/Analysis/Iterators.hpp"

#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

#include <mlir/IR/Builders.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/IR/OpImplementation.h>

#include <mlir/Interfaces/FunctionImplementation.h>

#include <llvm/Support/ErrorHandling.h>

#include <optional>
#include <variant>

namespace vast::core
{
    GRAPH_REGION_OP(ScopeOp);
} // namespace vast::core

//===----------------------------------------------------------------------===//
// TableGen generated logic.
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "vast/Dialect/Core/Core.cpp.inc"
