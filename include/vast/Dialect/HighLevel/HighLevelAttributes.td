// Copyright (c) 2021-present, Trail of Bits, Inc.

#ifndef VAST_DIALECT_HIGHLEVEL_IR_HIGHLEVELATTRIBUTES
#define VAST_DIALECT_HIGHLEVEL_IR_HIGHLEVELATTRIBUTES

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"

def APSIntAttr : Attr<CPred<"$_self.isa<::mlir::IntegerAttr>()">,
  "arbitrary integer attribute with sign">
{
  let storageType = [{ ::mlir::IntegerAttr }];
  let returnType = [{ ::llvm::APSInt }];
  let constBuilderCall = [{
    ::mlir::IntegerAttr::get($_builder.getContext(), $0)
  }];
  let convertFromStorage = [{
    llvm::APSInt($_self.getValue(), isSigned($_self.getType()));
  }];
}

def APFloatAttr : Attr<CPred<"$_self.isa<::mlir::FloatAttr>()">,
  "arbitrary floating-point attribute">
{
  let storageType = [{ ::mlir::FloatAttr }];
  let returnType = [{ ::llvm::APFloat }];
  let constBuilderCall = [{
    ::mlir::FloatAttr::get($_builder.getF64Type(), $0)
  }];
  let convertFromStorage = [{ $_self.getValue() }];
}

class HighLevel_Attr<string name, string attr_mnemonic>
    : AttrDef< HighLevel_Dialect, name >
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

def AnnotationAttr : NameAttr< "Annotation", "annotation" >;

#endif // VAST_DIALECT_HIGHLEVEL_IR_HIGHLEVELATTRIBUTES
