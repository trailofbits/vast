// Copyright (c) 2024, Trail of Bits, Inc.

#ifndef VAST_DIALECT_PARSER
#define VAST_DIALECT_PARSER

include "mlir/IR/OpBase.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"

def Parser_Dialect : Dialect {
  let name = "pr";
  let cppNamespace = "::vast::pr";

  let summary = "Dialect for abstracting parsering logic.";

  let description = [{
    Dialect for abstracting parsering logic.

    The goal of the dialect is to abstract the parsering logic
    for easier uniform parser recognition.
  }];

  let extraClassDeclaration = [{
    void registerTypes();
  }];

  let useDefaultTypePrinterParser = 1;

  let hasConstantMaterializer = 1;
}

include "vast/Dialect/Parser/Ops.td"
include "vast/Dialect/Parser/Types.td"

#endif // VAST_DIALECT_PARSER
