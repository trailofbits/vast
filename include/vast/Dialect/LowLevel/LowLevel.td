// Copyright (c) 2021-present, Trail of Bits, Inc.

#ifndef VAST_DIALECT_IR_LOWLEVEL
#define VAST_DIALECT_IR_LOWLEVEL

include "mlir/IR/OpBase.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

include "vast/Dialect/Core/Interfaces/SymbolInterface.td"

def LowLevel_Dialect : Dialect {
    let name = "ll";
    let summary = "A vast low-level dialect.";
    let description = [{
        This dialect serves as a bottom layer in VAST dialect tower.
        There should always exist a pass that lowers this dialect into LLVM Dialect.

        Work in progress - new operations are still being added and existing can be changed
        or removed.
    }];
    let cppNamespace = "::vast::ll";
    let extraClassDeclaration = [{
        void registerTypes();
        void registerAttributes();
    }];

    let dependentDialects = ["vast::core::CoreDialect"];
}

class LowLevel_Op< string mnemonic, list< Trait > traits = [] >
    : Op< LowLevel_Dialect, mnemonic, traits >;

include "LowLevelOps.td"

#endif // VAST_DIALECT_IR_LOWLEVEL
