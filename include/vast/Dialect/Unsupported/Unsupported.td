// Copyright (c) 2023-present, Trail of Bits, Inc.

#ifndef VAST_DIALECT_UNSUPPORTED
#define VAST_DIALECT_UNSUPPORTED

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"

include "mlir/Interfaces/DataLayoutInterfaces.td"
include "vast/Interfaces/DefaultDataLayoutTypeInterface.td"

include "vast/Dialect/Core/Interfaces/SymbolInterface.td"

def Unsupported_Dialect : Dialect {
    let name = "unsup";
    let cppNamespace = "::vast::unsup";

    let summary = "A vast unsupported dialect.";

    let description = [{
        This dialect defines a set of generic unsupported
        operation/types that can be used to lower AST Node
        that are yet not supported and can't be lowered by
        other dialects.
    }];

    let extraClassDeclaration = [{
        void registerTypes();
        void registerAttributes();
    }];

    let useDefaultTypePrinterParser = 1;
    let useDefaultAttributePrinterParser = 1;
}

class Unsupported_Op< string mnemonic, list< Trait > traits = [] >
    : Op< Unsupported_Dialect, mnemonic, traits >;

class UnsupportedType< string name, string _mnemonic, list<Trait> traits = [] >
    : TypeDef< Unsupported_Dialect, name, !listconcat(traits, [
      DataLayoutTypeInterface,
      DefaultDataLayoutTypeInterface
    ]) >
{
    let mnemonic = _mnemonic;
}

include "vast/Dialect/Unsupported/UnsupportedOps.td"
include "vast/Dialect/Unsupported/UnsupportedTypes.td"
include "vast/Dialect/Unsupported/UnsupportedAttributes.td"

#endif // VAST_DIALECT_UNSUPPORTED
