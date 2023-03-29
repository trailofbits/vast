// Copyright (c) 2022-present, Trail of Bits, Inc.

#include "vast/Util/Warnings.hpp"

#include "vast/Dialect/LowLevel/LowLevelDialect.hpp"
#include "vast/Dialect/LowLevel/LowLevelOps.hpp"

VAST_RELAX_WARNINGS
#include <mlir/IR/Builders.h>
VAST_UNRELAX_WARNINGS

#define GET_OP_CLASSES
#include "vast/Dialect/LowLevel/LowLevel.cpp.inc"

namespace vast::ll
{
    mlir::SuccessorOperands Br::getSuccessorOperands( unsigned idx )
    {
        VAST_CHECK( idx == 0, "ll::Br can have only one successor!" );
        return mlir::SuccessorOperands( getOperandsMutable() );
    }
} // namespace vast::ll
