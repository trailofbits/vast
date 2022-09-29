// Copyright (c) 2021-present, Trail of Bits, Inc.

#ifndef VAST_DIALECT_HIGHLEVEL_IR_HIGHLEVELATTRIBUTES
#define VAST_DIALECT_HIGHLEVEL_IR_HIGHLEVELATTRIBUTES

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"

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

class HighLevel_Attr<string name, string attr_mnemonic, list<Trait> traits = []>
    : AttrDef< HighLevel_Dialect, name, traits >
{
  let mnemonic = attr_mnemonic;
}

def BooleanAttr : HighLevel_Attr<"Boolean", "bool", [TypedAttrInterface] > {
  let summary = "An Attribute containing a boolean value";

  let description = [{
    An boolean attribute is a literal attribute that represents a boolean value.
  }];

  let parameters = (ins AttributeSelfTypeParameter<"">:$type, "bool":$value);

  let builders = [
    AttrBuilderWithInferredContext<(ins "Type":$type, "bool":$value), [{
      return $_get(type.getContext(), type, value);
    }]>
  ];

  let assemblyFormat = "`<` $value `>`";

  // let genVerifyDecl = 1;
}

def IntegerAttr : HighLevel_Attr<"Integer", "integer", [TypedAttrInterface] > {
  let summary = "An Attribute containing a integer value";

  let description = [{
    An integer attribute is a literal attribute that represents an integral
    value of the specified integer type.
  }];

  let parameters = (ins AttributeSelfTypeParameter<"">:$type, "llvm::APSInt":$value);

  let builders = [
    AttrBuilderWithInferredContext<(ins "Type":$type, "const llvm::APSInt &":$value), [{
      return $_get(type.getContext(), type, value);
    }]>
  ];

  let assemblyFormat = "`<` $value `>`";

  // let genVerifyDecl = 1;
}

def FloatAttr : HighLevel_Attr<"Float", "float", [TypedAttrInterface] > {
  let summary = "An Attribute containing a floating point value";

  let description = [{
    A float attribute is a literal attribute that represents a floating point
    value of the specified floating point type.
  }];

  let parameters = (ins AttributeSelfTypeParameter<"">:$type, "llvm::APFloat":$value);

  let builders = [
    AttrBuilderWithInferredContext<(ins "Type":$type, "const llvm::APFloat &":$value), [{
      return $_get(type.getContext(), type, value);
    }]>
  ];

  let assemblyFormat = "`<` $value `>`";

  // let genVerifyDecl = 1;
}

// Temporarily copied from builtin dialect, because llvm-15 does not have `TypedAttrInterface` yet.
def StringAttr : HighLevel_Attr<"String", "str", [TypedAttrInterface] > {
  let summary = "An Attribute containing a string";
  let description = [{
    Syntax:

    ```
    string-attribute ::= string-literal (`:` type)?
    ```

    A string attribute is an attribute that represents a string literal value.

    Examples:

    ```mlir
    "An important string"
    "string with a type" : !dialect.string
    ```
  }];
  let parameters = (ins StringRefParameter<"">:$value, AttributeSelfTypeParameter<"">:$type);
  let builders = [
    AttrBuilderWithInferredContext<(ins "const llvm::Twine &":$value, "Type":$type), [{
      llvm::SmallVector<char, 32> tempStr;
      return Base::get(type.getContext(), value.toStringRef(tempStr), type);
    }]>
  ];

  let assemblyFormat = "`<` $value `>`";
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
