// Copyright (c) 2021-present, Trail of Bits, Inc.

#ifndef VAST_DIALECT
#define VAST_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Vast dialect definition.
//===----------------------------------------------------------------------===//

def Vast_Dialect : Dialect {
    let name = "vast";
    let summary = "A verbose program analysis MLIR dialect.";
    let description = [{
        This dialect intends capture higher level constructs of C/C++
        for further program analysis.
    }];
    let cppNamespace = "::vast";
}

//===----------------------------------------------------------------------===//
// Base vast operation definition.
//===----------------------------------------------------------------------===//

class Vast_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<Vast_Dialect, mnemonic, traits>;

#endif // VAST_DIALECT