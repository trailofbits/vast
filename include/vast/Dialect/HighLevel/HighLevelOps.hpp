// Copyright (c) 2021-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <mlir/IR/BuiltinDialect.h>
#include <mlir/Interfaces/FunctionInterfaces.h>
#include <mlir/Interfaces/InferTypeOpInterface.h>
VAST_UNRELAX_WARNINGS

#include <gap/coro/generator.hpp>

#include "vast/Dialect/HighLevel/HighLevelAttributes.hpp"
#include "vast/Dialect/HighLevel/HighLevelDialect.hpp"
#include "vast/Dialect/HighLevel/HighLevelTypes.hpp"

#include "vast/Dialect/Core/SymbolTable.hpp"

#include "vast/Dialect/Core/Interfaces/SymbolInterface.hpp"
#include "vast/Dialect/Core/Interfaces/TypeDefinitionInterface.hpp"

#include "vast/Interfaces/TypeTraitExprInterface.hpp"
#include "vast/Interfaces/AST/DeclInterface.hpp"


#define GET_OP_CLASSES
#include "vast/Dialect/HighLevel/HighLevel.h.inc"

namespace vast::hl
{
    FuncOp getCallee(CallOp call);
} // namespace vast::hl
