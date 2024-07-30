// Copyright (c) 2024-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/OperationSupport.h>
VAST_RELAX_WARNINGS

#include "clang/AST/Stmt.h"

/// Include the generated interface declarations.
#include "vast/Interfaces/AST/StmtInterface.h.inc"
