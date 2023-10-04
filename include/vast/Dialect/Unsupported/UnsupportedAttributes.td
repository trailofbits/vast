// Copyright (c) 2023-present, Trail of Bits, Inc.

#ifndef VAST_DIALECT_IR_UNSUPPORTED_ATTRS
#define VAST_DIALECT_IR_UNSUPPORTED_ATTRS

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"

class Unsupported_Attr<string name, string attr_mnemonic, list<Trait> traits = []>
    : AttrDef< Unsupported_Dialect, name, traits >
{
  let mnemonic = attr_mnemonic;
}

def UnsupportedAttr
  : Unsupported_Attr< "Unsupported", "attr" >
{
    let parameters = (ins "::mlir::StringAttr":$spelling);
  let builders = [
    AttrBuilderWithInferredContext<(ins "::mlir::StringAttr":$name), [{
      return get(name.getContext(), name);
    }]>,
    AttrBuilder<(ins "::mlir::StringRef":$name), [{
      return get(mlir::StringAttr::get($_ctxt, name));
    }]>,
  ];
    let assemblyFormat = "$spelling";
}

#endif // VAST_DIALECT_IR_UNSUPPORTED_ATTRS
