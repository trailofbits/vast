// Copyright (c) 2023, Trail of Bits, Inc.

#ifndef VAST_DIALECT_CORE_COREATTRIBUTES
#define VAST_DIALECT_CORE_COREATTRIBUTES

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"

class Core_Attr<string name, string attr_mnemonic, list<Trait> traits = []>
    : AttrDef< Core_Dialect, name, traits >
{
  let mnemonic = attr_mnemonic;
}

def VoidAttr : Core_Attr<"Void", "void", [TypedAttrInterface]> {
  let summary = "Attribute to represent void value.";
  let description = [{
    The VoidAttr is used to return void from statements uniformly.
  }];

  let parameters = (ins AttributeSelfTypeParameter< "" >:$type);
  let assemblyFormat = [{}];
}

#endif // VAST_DIALECT_CORE_COREATTRIBUTES
