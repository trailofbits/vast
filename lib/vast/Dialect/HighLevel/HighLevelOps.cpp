// Copyright (c) 2021-present, Trail of Bits, Inc.

#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "vast/Dialect/HighLevel/HighLevel.hpp"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/Support/ErrorHandling.h"

namespace vast::hl
{
    void terminate_body(Builder &bld, Location loc)
    {
        bld.create< ScopeEndOp >(loc);
    }

    void ensure_terminator(Region *region, Builder &bld, Location loc)
    {
        if (region->empty())
            bld.createBlock(region);

        auto &block = region->back();
        if (!block.empty() && block.back().isKnownTerminator())
            return;
        bld.setInsertionPoint(&block, block.end());
        bld.create< ScopeEndOp >(loc);
    }
}

//===----------------------------------------------------------------------===//
// TableGen generated logic.
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "vast/Dialect/HighLevel/HighLevel.cpp.inc"
