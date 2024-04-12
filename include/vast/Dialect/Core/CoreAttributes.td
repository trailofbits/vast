// Copyright (c) 2023, Trail of Bits, Inc.

#ifndef VAST_DIALECT_CORE_COREATTRIBUTES
#define VAST_DIALECT_CORE_COREATTRIBUTES

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"

include "mlir/IR/EnumAttr.td"

class Core_Attr<string name, string attr_mnemonic, list<Trait> traits = []>
    : AttrDef< Core_Dialect, name, traits >
{
  let mnemonic = attr_mnemonic;
}

class Core_I32Enum<string name, string description, list<I32EnumAttrCase> cases>
    : I32EnumAttr<name, description, cases>
{
  let genSpecializedAttr = 0;
  let cppNamespace = "::vast::core";
}
class Core_I32EnumAttr<string mnemonic, Core_I32Enum enumInfo>
  : EnumAttr<Core_Dialect, enumInfo, mnemonic>
{
  let assemblyFormat = "`<` $value `>`";
}

class APSIntParameter<string desc> :
    AttrOrTypeParameter<"::llvm::APSInt", desc> {
  let comparator = [{
    $_lhs.getBitWidth() == $_rhs.getBitWidth() &&
    $_lhs.isSigned() == $_rhs.isSigned() &&
    $_lhs == $_rhs
  }];
}

def BooleanAttr : Core_Attr<"Boolean", "bool",
  [ConstantLikeAttrTrait, TypedAttrInterface]
> {
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

def IntegerAttr : Core_Attr<"Integer", "integer",
  [ConstantLikeAttrTrait, TypedAttrInterface]
> {
  let summary = "An Attribute containing a integer value";

  let description = [{
    An integer attribute is a literal attribute that represents an integral
    value of the specified integer type.
  }];

  let parameters = (ins AttributeSelfTypeParameter<"">:$type, APSIntParameter<"">:$value);

  let builders = [
    AttrBuilderWithInferredContext<(ins "Type":$type, "const llvm::APSInt &":$value), [{
      return $_get(type.getContext(), type, value);
    }]>
  ];

  let assemblyFormat = "`<` $value `>`";

  // let genVerifyDecl = 1;
}

def FloatAttr : Core_Attr<"Float", "float",
  [ConstantLikeAttrTrait, TypedAttrInterface]
> {
  let summary = "An Attribute containing a floating point value";

  let description = [{
    A float attribute is a literal attribute that represents a floating point
    value of the specified floating point type.
  }];

  let parameters = (ins AttributeSelfTypeParameter<"">:$type, APFloatParameter<"">:$value);

  let builders = [
    AttrBuilderWithInferredContext<(ins "Type":$type, "const llvm::APFloat &":$value), [{
      return $_get(type.getContext(), type, value);
    }]>
  ];

  let assemblyFormat = "`<` $value `>`";

  // let genVerifyDecl = 1;
}

def VoidAttr : Core_Attr<"Void", "void",
  [ConstantLikeAttrTrait, TypedAttrInterface]
> {
  let summary = "Attribute to represent void value.";
  let description = [{
    The VoidAttr is used to return void from statements uniformly.
  }];

  let parameters = (ins AttributeSelfTypeParameter< "" >:$type);
  let assemblyFormat = [{}];
}

def C : I32EnumAttrCase<"C", 1, "c">;
def CXX : I32EnumAttrCase<"CXX", 2, "cxx">;

def SourceLanguage : I32EnumAttr<"SourceLanguage", "Source language", [
  C, CXX
]> {
  let cppNamespace = "::vast::core";
  let genSpecializedAttr = 0;
}

def SourceLanguageAttr : EnumAttr< Core_Dialect, SourceLanguage, "lang" > {
  let summary = "Module source language";
  let description = [{
    Represents the source language used to generate the module.

    Example:
    ```
    // Module compiled from C.
    module attributes {vast.core.lang = vast.core.lang<c>} {}
    // Module compiled from C++.
    module attributes {vast.core.lang = vast.core.lang<cxx>} {}
    ```
  }];
}

include "vast/Dialect/Core/Linkage.td"

#endif // VAST_DIALECT_CORE_COREATTRIBUTES
