// Copyright (c) 2021-present, Trail of Bits, Inc.

#include "vast/Dialect/HighLevel/HighLevel.hpp"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/OpImplementation.h"

namespace vast::hl
{
    using builder = mlir::OpBuilder;

    void FuncOp::build(builder &bld, mlir::OperationState &st, llvm::StringRef name)
    {
        st.addRegion();
        st.addAttribute(mlir::SymbolTable::getSymbolAttrName(), bld.getStringAttr(name));
    }
}

//===----------------------------------------------------------------------===//
// TableGen generated logic.
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "vast/Dialect/HighLevel/HighLevel.cpp.inc"