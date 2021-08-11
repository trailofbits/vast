// Copyright (c) 2021-present, Trail of Bits, Inc.

#ifndef VAST_DIALECT_HIGHLEVEL_IR_HIGHLEVELOPS
#define VAST_DIALECT_HIGHLEVEL_IR_HIGHLEVELOPS

def HighLevel_FuncOp : HighLevel_Op< "func", [] >
{
    let summary = "VAST high-level function.";

    let description = [{
        VAST high-level function
    }];

    // let arguments =

    let regions = (region AnyRegion:$body);

    let skipDefaultBuilders = 1;

    let builders = [
        OpBuilderDAG< (
            ins "llvm::StringRef":$name
        ) >
    ];

    // let printer =

    // let parser =
}

#endif // VAST_DIALECT_HIGHLEVEL_IR_HIGHLEVELOPS
