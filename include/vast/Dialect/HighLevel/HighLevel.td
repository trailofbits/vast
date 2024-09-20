// Copyright (c) 2021-present, Trail of Bits, Inc.

#ifndef VAST_DIALECT_HIGHLEVEL_IR_HIGHLEVEL
#define VAST_DIALECT_HIGHLEVEL_IR_HIGHLEVEL

include "mlir/IR/OpBase.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"

include "vast/Dialect/Core/Interfaces/SymbolInterface.td"

include "vast/Dialect/Core/Utils.td"

def HighLevel_Dialect : Dialect {
    let name = "hl";
    let cppNamespace = "::vast::hl";

    let summary = "A high-level verbose program analysis MLIR dialect.";

    let description = [{
        This dialect intends capture highevel constructs of C/C++
        for further program analysis.
    }];

    let extraClassDeclaration = [{
        void registerTypes();
        void registerAttributes();
    }];

    let useDefaultTypePrinterParser = 1;
    let useDefaultAttributePrinterParser = 1;

    let hasConstantMaterializer = 1;

    let dependentDialects = ["vast::core::CoreDialect"];

    let usePropertiesForAttributes = 0;
}

class HighLevel_Op< string mnemonic, list< Trait > traits = [] >
    : Op< HighLevel_Dialect, mnemonic, traits >;

include "HighLevelAttributes.td"
include "HighLevelTypes.td"
include "HighLevelVar.td"
include "HighLevelOps.td"
include "HighLevelOpsCxx.td"
include "HighLevelCF.td"

#endif // VAST_DIALECT_HIGHLEVEL_IR_HIGHLEVEL
