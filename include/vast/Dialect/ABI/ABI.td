// Copyright (c) 2021-present, Trail of Bits, Inc.

#ifndef VAST_DIALECT_IR_ABI
#define VAST_DIALECT_IR_ABI

include "mlir/IR/OpBase.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/RegionKindInterface.td"

include "vast/Dialect/Core/Interfaces/SymbolInterface.td"

def ABI_Dialect : Dialect {
    let name = "abi";
    let summary = "A vast ABI dialect.";
    let description = [{
        Dialect provides operations to describe how arguments and return values
        are transformed to better model target abi.
    }];
    let cppNamespace = "::vast::abi";
    let extraClassDeclaration = [{
        void registerTypes();
        void registerAttributes();
    }];

    let dependentDialects = ["vast::core::CoreDialect"];
}

class ABI_Op< string mnemonic, list< Trait > traits = [] >
    : Op< ABI_Dialect, mnemonic, traits >;

include "ABIOps.td"

#endif // VAST_DIALECT_IR_ABI
