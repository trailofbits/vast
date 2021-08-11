// Copyright (c) 2021-present, Trail of Bits, Inc.

#ifndef VAST_DIALECT_HIGHLEVEL_IR_HIGHLEVELBASE
#define VAST_DIALECT_HIGHLEVEL_IR_HIGHLEVELBASE

include "mlir/IR/OpBase.td"

def HighLevel_Dialect : Dialect {
    let name = "highlevel";
    let summary = "A high-level verbose program analysis MLIR dialect.";
    let description = [{
        This dialect intends capture highevel constructs of C/C++
        for further program analysis.
    }];
    let cppNamespace = "::vast::hl";
}

#endif // VAST_DIALECT_HIGHLEVEL_IR_HIGHLEVELBASE