// Copyright (c) 2021-present, Trail of Bits, Inc.

#ifndef VAST_DIALECT_HIGHLEVEL_IR_HIGHLEVEL
#define VAST_DIALECT_HIGHLEVEL_IR_HIGHLEVEL

include "mlir/IR/OpBase.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/RegionKindInterface.td"

include "vast/Interfaces/SymbolInterface.td"

def HighLevel_Dialect : Dialect {
    let name = "hl";
    let summary = "A high-level verbose program analysis MLIR dialect.";
    let description = [{
        This dialect intends capture highevel constructs of C/C++
        for further program analysis.
    }];
    let cppNamespace = "::vast::hl";
    let extraClassDeclaration = [{
        void registerTypes();
        void registerAttributes();
    }];

    let hasConstantMaterializer = 1;
}

class HighLevel_Op< string mnemonic, list< Trait > traits = [] >
    : Op< HighLevel_Dialect, mnemonic, traits >;

def HasOneBlock : CPred<"$_self.hasOneBlock()">;

include "HighLevelAttributes.td"
include "HighLevelTypes.td"
include "HighLevelVar.td"
include "HighLevelOps.td"
include "HighLevelCF.td"

#endif // VAST_DIALECT_HIGHLEVEL_IR_HIGHLEVEL
