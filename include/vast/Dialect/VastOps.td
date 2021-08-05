// Copyright (c) 2021-present, Trail of Bits, Inc.

#ifndef VAST_OPS
#define VAST_OPS

include "VastDialect.td"

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Vast_VastFuncOp : Vast_Op< "func", [] >
{
    let summary = "VAST high-level function.";

    let description = [{
        VAST function
    }];

    // let arguments =

    let regions = (region AnyRegion:$body);

    let skipDefaultBuilders = 1;

    let builders = [
        OpBuilderDAG< (
            ins "llvm::StringRef":$name
        ) >
    ];

    // let verifier =


    // let printer =

    // let parser =
}

#endif // VAST_OPS