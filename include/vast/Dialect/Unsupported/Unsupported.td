// Copyright (c) 2023-present, Trail of Bits, Inc.

#ifndef VAST_DIALECT_UNSUPPORTED
#define VAST_DIALECT_UNSUPPORTED

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"

include "vast/Interfaces/SymbolInterface.td"

def Unsupported_Dialect : Dialect {
    let name = "unsupported";
    let cppNamespace = "::vast::us";

    let summary = "A vast unsupported dialect.";

    let description = [{
        This dialect defines a set of generic unsupported
        operation/types that can be used to lower AST Node
        that are yet not supported and can't be lowered by
        other dialects.
    }];

    let extraClassDeclaration = [{
        void registerTypes();
    }];

    let useFoldAPI = kEmitFoldAdaptorFolder;
    let useDefaultTypePrinterParser = 1;
}

class Unsupported_Op< string mnemonic, list< Trait > traits = [] >
    : Op< Unsupported_Dialect, mnemonic, traits >;

 class Unsupported_Type< string name, string typeMnemonic, list<Trait> traits = [] >
    : TypeDef< Unsupported_Dialect, name, traits> {
    let mnemonic = typeMnemonic;
 }

include "vast/Dialect/Unsupported/UnsupportedOps.td"
include "vast/Dialect/Unsupported/UnsupportedTypes.td"

#endif // VAST_DIALECT_UNSUPPORTED
