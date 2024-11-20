// Copyright (c) 2024, Trail of Bits, Inc.

#ifndef VAST_INTERFACES_OPERATION_INTERFACES
#define VAST_INTERFACES_OPERATION_INTERFACES

include "mlir/IR/OpBase.td"

include "vast/Dialect/Core/CoreTraits.td"
include "vast/Dialect/Core/Interfaces/Common.td"

def Core_ArithBinOp : Core_OpInterface< "ArithBinOpInterface" > {
    let description = [{
        This interface describes an operation that is arithmetic binary operation.
    }];
}

#endif // VAST_INTERFACES_OPERATION_INTERFACES
