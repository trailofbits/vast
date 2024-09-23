// Copyright (c) 2023, Trail of Bits, Inc.

#ifndef VAST_DIALECT_CORE_COREATTRIBUTES
#define VAST_DIALECT_CORE_COREATTRIBUTES

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "vast/Dialect/Core/Interfaces/SymbolRefInterface.td"

include "mlir/IR/EnumAttr.td"

class Core_Attr< string name, string attr_mnemonic, list<Trait> traits = [] >
    : AttrDef< Core_Dialect, name, traits >
{
  let mnemonic = attr_mnemonic;
}

class Core_I32Enum< string name, string description, list<I32EnumAttrCase> cases >
    : I32EnumAttr< name, description, cases >
{
  let genSpecializedAttr = 0;
  let cppNamespace = "::vast::core";
}

class Core_I32EnumAttr< string mnemonic, Core_I32Enum enumInfo >
  : EnumAttr< Core_Dialect, enumInfo, mnemonic >
{
  let assemblyFormat = "`<` $value `>`";
}

class Core_APSIntParameter< string desc >
    : AttrOrTypeParameter< "::llvm::APSInt", desc >
{
  let comparator = [{
    $_lhs.getBitWidth() == $_rhs.getBitWidth() &&
    $_lhs.isSigned() == $_rhs.isSigned() &&
    $_lhs == $_rhs
  }];
}

def Core_BooleanAttr : Core_Attr< "Boolean", "bool",
  [Core_ConstantLikeAttrTrait, TypedAttrInterface]
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

def Core_IntegerAttr : Core_Attr< "Integer", "integer",
  [Core_ConstantLikeAttrTrait, TypedAttrInterface]
> {
  let summary = "An Attribute containing a integer value";

  let description = [{
    An integer attribute is a literal attribute that represents an integral
    value of the specified integer type.
  }];

  let parameters = (ins AttributeSelfTypeParameter<"">:$type, Core_APSIntParameter<"">:$value);

  let builders = [
    AttrBuilderWithInferredContext<(ins "Type":$type, "const llvm::APSInt &":$value), [{
      return $_get(type.getContext(), type, value);
    }]>
  ];

  let assemblyFormat = "`<` $value `>`";

  // let genVerifyDecl = 1;
}

def Core_FloatAttr : Core_Attr< "Float", "float",
  [Core_ConstantLikeAttrTrait, TypedAttrInterface]
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

def Core_VoidAttr : Core_Attr< "Void", "void",
  [Core_ConstantLikeAttrTrait, TypedAttrInterface]
> {
  let summary = "Attribute to represent void value.";
  let description = [{
    The VoidAttr is used to return void from statements uniformly.
  }];

  let parameters = (ins AttributeSelfTypeParameter< "" >:$type);
  let assemblyFormat = [{}];
}

def Core_SourceLanguage : I32EnumAttr< "SourceLanguage", "Source language", [
  I32EnumAttrCase<"C", 1, "c">,
  I32EnumAttrCase<"CXX", 2, "cxx">
] > {
  let cppNamespace = "::vast::core";
  let genSpecializedAttr = 0;
}

def Core_SourceLanguageAttr : EnumAttr< Core_Dialect, Core_SourceLanguage, "lang" > {
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

//
// Storage and context attributes
//

class Core_StorageClassAttr< string name, int val >
  : I64EnumAttrCase< name, val >
{}

class Core_StorageClassList< string name, string summary, list<Core_StorageClassAttr> cases >
  : I64EnumAttr< name, summary, cases >
{}

def Core_SC_None          : Core_StorageClassAttr< "sc_none", 0 >;
def Core_SC_Auto          : Core_StorageClassAttr< "sc_auto", 1 >;
def Core_SC_Static        : Core_StorageClassAttr< "sc_static", 2 >;
def Core_SC_Extern        : Core_StorageClassAttr< "sc_extern", 3 >;
def Core_SC_PrivateExtern : Core_StorageClassAttr< "sc_private_extern", 4 >;
def Core_SC_Register      : Core_StorageClassAttr< "sc_register", 5 >;

let cppNamespace = "::vast::core" in
def Core_StorageClass : Core_StorageClassList< "StorageClass", "storage class", [
  Core_SC_None,
  Core_SC_Auto,
  Core_SC_Static,
  Core_SC_Extern,
  Core_SC_PrivateExtern,
  Core_SC_Register
] >;

// thread storage class

class Core_TSCAttr< string name, int val > : I64EnumAttrCase< name, val > {}

class Core_TSCList< string name, string summary, list<Core_TSCAttr> cases >
  : I64EnumAttr< name, summary, cases >
{}

def Core_TSC_None       : Core_TSCAttr< "tsc_none", 0 >;
def Core_TSC_GNU_Thread : Core_TSCAttr< "tsc_gnu_thread", 1 >;
def Core_TSC_CXX_Thread : Core_TSCAttr< "tsc_cxx_thread", 2 >;
def Core_TSC_C_Thread   : Core_TSCAttr< "tsc_c_thread", 3 >;

let cppNamespace = "::vast::core" in
def Core_ThreadStorage : Core_TSCList< "TSClass", "thread storage class",
[
  Core_TSC_None,
  Core_TSC_GNU_Thread,
  Core_TSC_CXX_Thread,
  Core_TSC_C_Thread
] >;

// Storage Duration

class Core_StorageDurationAttr< string name, int val > : I64EnumAttrCase< name, val > {}

class Core_StorageDurationList< string name, string summary, list<Core_StorageDurationAttr> cases >
  : I64EnumAttr< name, summary, cases >
{}

def Core_SD_FullExpression : Core_StorageDurationAttr< "sd_none", 0 >;
def Core_SD_Automatic      : Core_StorageDurationAttr< "sd_automatic", 1 >;
def Core_SD_Thread 	       : Core_StorageDurationAttr< "sd_thread", 2 >;
def Core_SD_Static         : Core_StorageDurationAttr< "sd_static", 3 >;
def Core_SD_Dynamic        : Core_StorageDurationAttr< "sd_dynamic", 4 >;

let cppNamespace = "::vast::core" in
def Core_StorageDuration
  : Core_StorageDurationList< "StorageDuration", "storage duration",
[
  Core_SD_FullExpression,
  Core_SD_Automatic,
  Core_SD_Thread,
  Core_SD_Static,
  Core_SD_Dynamic
] >;

class Core_StorageDurationImpl
{
  code storageDurationImpl = [{
    bool hasLocalStorage();

    bool isLocalVarDecl();

    bool isStaticLocal();

    bool hasExternalStorage();

    bool hasGlobalStorage();

    StorageDuration getStorageDuration();
  }];
}

// declaration context kinds
class Core_DeclContextAttr< string name, int val > : I64EnumAttrCase< name, val > {}

class Core_DeclContextList< string name, string summary, list<Core_DeclContextAttr> cases >
  : I64EnumAttr< name, summary, cases >
{}

def Core_DC_Function        : Core_DeclContextAttr< "dc_function", 0 >;
def Core_DC_Method          : Core_DeclContextAttr< "dc_method", 1 >;
def Core_DC_Block           : Core_DeclContextAttr< "dc_block", 2 >;
def Core_DC_Capture         : Core_DeclContextAttr< "dc_capture", 3 >;
def Core_DC_TranslationUnit : Core_DeclContextAttr< "dc_translation_unit", 4 >;
def Core_DC_Record          : Core_DeclContextAttr< "dc_record", 5 >;
def Core_DC_Enum            : Core_DeclContextAttr< "dc_enum", 6 >;
def Core_DC_Namespace       : Core_DeclContextAttr< "dc_namespace", 7 >;

let cppNamespace = "::vast::core" in
def Core_DeclContextKind
  : Core_DeclContextList< "DeclContextKind", "declaration context kind",
[
  Core_DC_Function,
  Core_DC_Method,
  Core_DC_Block,
  Core_DC_Capture,
  Core_DC_TranslationUnit,
  Core_DC_Record,
  Core_DC_Enum,
  Core_DC_Namespace
] >;

class Core_DeclContextImpl
{
  code declContextImpl = [{
    DeclContextKind getDeclContextKind();

    bool isStaticDataMember();

    bool isInFileContext();

    bool isInFunctionOrMethodContext();

    bool isInRecordContext();

    bool isFileVarDecl();
  }];
}

class Core_StorageSpecifiers :
  Core_StorageDurationImpl,
  Core_DeclContextImpl
{
  code storageSpecifiersImpl =
    storageDurationImpl #
    declContextImpl;
}


#endif // VAST_DIALECT_CORE_COREATTRIBUTES
