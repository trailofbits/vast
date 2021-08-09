// Copyright (c) 2021-present, Trail of Bits, Inc.

#include "vast/Dialect/VastOps.hpp"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/SymbolTable.h"
#include "vast/Dialect/VastDialect.hpp"
#include "mlir/IR/OpImplementation.h"

namespace vast::hl
{
    using builder = mlir::OpBuilder;

    void VastFuncOp::build(builder &bld, mlir::OperationState &st, llvm::StringRef name)
    {
        st.addRegion();
        st.addAttribute(mlir::SymbolTable::getSymbolAttrName(), bld.getStringAttr(name));
    }

} // namespace vast::hl

//===----------------------------------------------------------------------===//
// TableGen generated logic.
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "vast/Dialect/VastOps.cpp.inc"