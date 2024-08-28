// Copyright (c) 2021-present, Trail of Bits, Inc.

#ifndef VAST_DIALECT_HIGHLEVEL_IR_HIGHLEVELVAR
#define VAST_DIALECT_HIGHLEVEL_IR_HIGHLEVELVAR

include "vast/Dialect/Core/CoreTraits.td"

class HighLevel_StorageClassAttr< string name, int val >
  : I64EnumAttrCase< name, val >
{}

class HighLevel_StorageClassList< string name, string summary, list<HighLevel_StorageClassAttr> cases >
  : I64EnumAttr< name, summary, cases >
{}

def HighLevel_SC_None          : HighLevel_StorageClassAttr< "sc_none", 0 >;
def HighLevel_SC_Auto          : HighLevel_StorageClassAttr< "sc_auto", 1 >;
def HighLevel_SC_Static        : HighLevel_StorageClassAttr< "sc_static", 2 >;
def HighLevel_SC_Extern        : HighLevel_StorageClassAttr< "sc_extern", 3 >;
def HighLevel_SC_PrivateExtern : HighLevel_StorageClassAttr< "sc_private_extern", 4 >;
def HighLevel_SC_Register      : HighLevel_StorageClassAttr< "sc_register", 5 >;

let cppNamespace = "::vast::hl" in
def HighLevel_StorageClass : HighLevel_StorageClassList< "StorageClass", "storage class", [
  HighLevel_SC_None,
  HighLevel_SC_Auto,
  HighLevel_SC_Static,
  HighLevel_SC_Extern,
  HighLevel_SC_PrivateExtern,
  HighLevel_SC_Register
] >;

class HighLevel_StorageClassImpl
{
  code storageClassImpl = [{
    constexpr static auto storage_class = "storageClass";

    void setStorageClass(StorageClass spec) { setAttr< StorageClassAttr >(storage_class, spec); }
  }];
}

// thread storage class

class HighLevel_TSCAttr< string name, int val > : I64EnumAttrCase< name, val > {}

class HighLevel_TSCList< string name, string summary, list<HighLevel_TSCAttr> cases >
  : I64EnumAttr< name, summary, cases >
{}

def HighLevel_TSC_None       : HighLevel_TSCAttr< "tsc_none", 0 >;
def HighLevel_TSC_GNU_Thread : HighLevel_TSCAttr< "tsc_gnu_thread", 1 >;
def HighLevel_TSC_CXX_Thread : HighLevel_TSCAttr< "tsc_cxx_thread", 2 >;
def HighLevel_TSC_C_Thread   : HighLevel_TSCAttr< "tsc_c_thread", 3 >;

let cppNamespace = "::vast::hl" in
def HighLevel_ThreadStorage : HighLevel_TSCList< "TSClass", "thread storage class",
[
  HighLevel_TSC_None,
  HighLevel_TSC_GNU_Thread,
  HighLevel_TSC_CXX_Thread,
  HighLevel_TSC_C_Thread
] >;

class HighLevel_ThreadStorageClassImpl
{
  code threadStorageClassImpl = [{
    constexpr static auto thread_storage_class = "threadStorageClass";

    void setThreadStorageClass(TSClass spec) { setAttr< TSClassAttr >(thread_storage_class, spec); }
  }];
}

// Storage Duration

class HighLevel_StorageDurationAttr< string name, int val > : I64EnumAttrCase< name, val > {}

class HighLevel_StorageDurationList< string name, string summary, list<HighLevel_StorageDurationAttr> cases >
  : I64EnumAttr< name, summary, cases >
{}

def HighLevel_SD_FullExpression : HighLevel_StorageDurationAttr< "sd_none", 0 >;
def HighLevel_SD_Automatic      : HighLevel_StorageDurationAttr< "sd_automatic", 1 >;
def HighLevel_SD_Thread 	      : HighLevel_StorageDurationAttr< "sd_thread", 2 >;
def HighLevel_SD_Static         : HighLevel_StorageDurationAttr< "sd_static", 3 >;
def HighLevel_SD_Dynamic        : HighLevel_StorageDurationAttr< "sd_dynamic", 4 >;

let cppNamespace = "::vast::hl" in
def HighLevel_StorageDuration
  : HighLevel_StorageDurationList< "StorageDuration", "storage duration",
[
  HighLevel_SD_FullExpression,
  HighLevel_SD_Automatic,
  HighLevel_SD_Thread,
  HighLevel_SD_Static,
  HighLevel_SD_Dynamic
] >;

class HighLevel_StorageDurationImpl
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
class HighLevel_DeclContextAttr< string name, int val > : I64EnumAttrCase< name, val > {}

class HighLevel_DeclContextList< string name, string summary, list<HighLevel_DeclContextAttr> cases >
  : I64EnumAttr< name, summary, cases >
{}

def HighLevel_DC_Function        : HighLevel_DeclContextAttr< "dc_function", 0 >;
def HighLevel_DC_Method          : HighLevel_DeclContextAttr< "dc_method", 1 >;
def HighLevel_DC_Block           : HighLevel_DeclContextAttr< "dc_block", 2 >;
def HighLevel_DC_Capture         : HighLevel_DeclContextAttr< "dc_capture", 3 >;
def HighLevel_DC_TranslationUnit : HighLevel_DeclContextAttr< "dc_translation_unit", 4 >;
def HighLevel_DC_Record          : HighLevel_DeclContextAttr< "dc_record", 5 >;
def HighLevel_DC_Enum            : HighLevel_DeclContextAttr< "dc_enum", 6 >;
def HighLevel_DC_Namespace       : HighLevel_DeclContextAttr< "dc_namespace", 7 >;

let cppNamespace = "::vast::hl" in
def HighLevel_DeclContextKind
  : HighLevel_DeclContextList< "DeclContextKind", "declaration context kind",
[
  HighLevel_DC_Function,
  HighLevel_DC_Method,
  HighLevel_DC_Block,
  HighLevel_DC_Capture,
  HighLevel_DC_TranslationUnit,
  HighLevel_DC_Record,
  HighLevel_DC_Enum,
  HighLevel_DC_Namespace
] >;

class HighLevel_DeclContextImpl
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

class HighLevel_StorageSpecifiers :
  HighLevel_StorageClassImpl,
  HighLevel_ThreadStorageClassImpl,
  HighLevel_StorageDurationImpl,
  HighLevel_DeclContextImpl
{
  code storageSpecifiersImpl =
    storageClassImpl #
    threadStorageClassImpl #
    storageDurationImpl #
    declContextImpl;
}

// Variable Operation
def HighLevel_VarDeclOp : HighLevel_Op< "var", [VarSymbol] >
  , HighLevel_StorageSpecifiers
{
  let summary = "VAST variable declaration";
  let description = [{ VAST variable declaration }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    OptionalAttr<HighLevel_StorageClass>:$storageClass,
    OptionalAttr<HighLevel_ThreadStorage>:$threadStorageClass
  );

  let results = (outs AnyType:$result);
  let regions = (region
    AnyRegion:$initializer,
    AnyRegion:$allocation_size
  );

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins
      "Type":$type,
      "llvm::StringRef":$sym_name,
      CArg< "maybe_builder_callback_ref", "std::nullopt" >:$initBuilder,
      CArg< "maybe_builder_callback_ref", "std::nullopt" >:$allocaBuilder
    )>
  ];

  let assemblyFormat = [{
    $sym_name attr-dict ($storageClass^)? ($threadStorageClass^)? `:` type($result)
      (`=` $initializer^)?
      (`allocation_size` $allocation_size^)?
  }];

  let extraClassDeclaration = [{
    template< typename Attr, typename Value >
    void setAttr(llvm::StringRef name, Value value) {
      (*this)->setAttr(name, Attr::get(getContext(), value));
    }
  }] # storageSpecifiersImpl;
}

#endif // VAST_DIALECT_HIGHLEVEL_IR_HIGHLEVELVAR
