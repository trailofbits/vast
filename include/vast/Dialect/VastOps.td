// Copyright (c) 2021-present, Trail of Bits, Inc.

#ifndef VAST_OPS
#define VAST_OPS

include "VastDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Vast_TestOp : Vast_Op<"test", [NoSideEffect, SameOperandsAndResultType]> {
    let summary = "Test operation.";
    let description = [{
        The `vast.test` operation.
    }];
    let arguments = (ins I32:$input);
    let results = (outs I32:$res);

    let assemblyFormat = [{
        $input attr-dict `:` type($input)
    }];
}

#endif // VAST_OPS