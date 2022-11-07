// Copyright (c) 2022-present, Trail of Bits, Inc.

#ifndef VAST_DIALECT_IR_ABIOPS
#define VAST_DIALECT_IR_ABIOPS

include "mlir/Interfaces/CallInterfaces.td"
include "mlir/IR/BuiltinAttributes.td"
include "vast/Interfaces/SymbolInterface.td"

def Forward
    : ABI_Op< "forward", [NoSideEffect] >
    , Arguments<(ins AnyType:$value)>
    , Results<(outs AnyType:$result)>
{
    let summary = "Pass argument as it is.";
    let description = [{ Pass argument as it is. }];
}

#endif // VAST_DIALECT_IR_ABIOPS
