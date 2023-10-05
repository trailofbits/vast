// Copyright (c) 2021-present, Trail of Bits, Inc.

#ifndef VAST_DIALECT_HIGHLEVEL_IR_HIGHLEVELATTRIBUTES
#define VAST_DIALECT_HIGHLEVEL_IR_HIGHLEVELATTRIBUTES

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"

class HighLevel_Attr<string name, string attr_mnemonic, list<Trait> traits = []>
    : AttrDef< HighLevel_Dialect, name, traits >
{
  let mnemonic = attr_mnemonic;
}

class NameAttr<string name, string attr_mnemonic>
  : HighLevel_Attr< name, attr_mnemonic >
{
  let parameters = (ins "::mlir::StringAttr":$name);
  let builders = [
    AttrBuilderWithInferredContext<(ins "::mlir::StringAttr":$name), [{
      return get(name.getContext(), name);
    }]>,
    AttrBuilder<(ins "::mlir::StringRef":$name), [{
      return get(mlir::StringAttr::get($_ctxt, name));
    }]>,
  ];

  let assemblyFormat = "$name";
}

def AnnotationAttr    : NameAttr< "Annotation", "annotation" >;
def SectionAttr       : NameAttr< "Section", "section" >;
def LoaderUninitAttr  : HighLevel_Attr< "LoaderUninitialized", "loader_uninitialized" >;
def NoInstrumentAttr  : HighLevel_Attr< "NoInstrumentFunction", "no_instrument_function" >;
def PackedAttr        : HighLevel_Attr< "Packed", "packed" >;
def WarnUnusedResAttr : HighLevel_Attr< "WarnUnusedResult", "warn_unused_res" >;

#endif // VAST_DIALECT_HIGHLEVEL_IR_HIGHLEVELATTRIBUTES
