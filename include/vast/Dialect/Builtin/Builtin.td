// Copyright (c) 2024-present, Trail of Bits, Inc.

#ifndef VAST_DIALECT_BUILTIN
#define VAST_DIALECT_BUILTIN

include "mlir/IR/OpBase.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"

def Builtin_Dialect : Dialect {
  let name = "hlbi";
  let cppNamespace = "::vast::hlbi";

  let summary = "Dialect for representing compiler builtin constructs";

  let description = [{
    TODO
  }];

  let extraClassDeclaration = [{
      void registerTypes();
  }];

  let useDefaultTypePrinterParser = 1;

  let hasConstantMaterializer = 1;
}

class Builtin_Op< string mnemonic, list< Trait > traits = [] >
  : Op< Builtin_Dialect, mnemonic, traits >;

include "vast/Dialect/Builtin/Types.td"
include "vast/Dialect/Builtin/Ops.td"

#endif // VAST_DIALECT_BUILTIN
