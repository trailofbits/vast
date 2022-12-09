// Copyright (c) 2022-present, Trail of Bits, Inc.

#ifndef VAST_DIALECT_IR_ABIOPS
#define VAST_DIALECT_IR_ABIOPS

include "mlir/Interfaces/CallInterfaces.td"
include "mlir/IR/BuiltinAttributes.td"
include "vast/Interfaces/SymbolInterface.td"

def DirectOp
    : ABI_Op< "direct", [NoSideEffect] >
    , Arguments<(ins AnyType:$value)>
    , Results<(outs Variadic<AnyType>:$result)>
{
    let summary = "Pass value directly - usually means by register";
    let description = [{ Pass value directly - usually means by register. }];
}

#endif // VAST_DIALECT_IR_ABIOPS
