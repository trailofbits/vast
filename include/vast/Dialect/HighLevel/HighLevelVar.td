// Copyright (c) 2021-present, Trail of Bits, Inc.

#ifndef VAST_DIALECT_HIGHLEVEL_IR_HIGHLEVELVAR
#define VAST_DIALECT_HIGHLEVEL_IR_HIGHLEVELVAR

include "vast/Dialect/Core/CoreTraits.td"

class StorageClassAttr< string name, int val >
  : I64EnumAttrCase< name, val >
{}

class StorageClassList< string name, string summary, list<StorageClassAttr> cases >
  : I64EnumAttr< name, summary, cases >
{}

def SC_None          : StorageClassAttr< "sc_none", 0 >;
def SC_Auto          : StorageClassAttr< "sc_auto", 1 >;
def SC_Static        : StorageClassAttr< "sc_static", 2 >;
def SC_Extern        : StorageClassAttr< "sc_extern", 3 >;
def SC_PrivateExtern : StorageClassAttr< "sc_private_extern", 4 >;
def SC_Register      : StorageClassAttr< "sc_register", 5 >;

let cppNamespace = "::vast::hl" in
def StorageClass : StorageClassList< "StorageClass", "storage class", [
  SC_None, SC_Auto, SC_Static, SC_Extern, SC_PrivateExtern, SC_Register
] >;

class StorageClassImpl
{
  code storageClassImpl = [{
    constexpr static auto storage_class = "storageClass";

    void setStorageClass(StorageClass spec) { setAttr< StorageClassAttr >(storage_class, spec); }
  }];
}

// thread storage class

class TSCAttr< string name, int val > : I64EnumAttrCase< name, val > {}

class TSCList< string name, string summary, list<TSCAttr> cases >
  : I64EnumAttr< name, summary, cases >
{}

def TSC_None       : TSCAttr< "tsc_none", 0 >;
def TSC_GNU_Thread : TSCAttr< "tsc_gnu_thread", 1 >;
def TSC_CXX_Thread : TSCAttr< "tsc_cxx_thread", 2 >;
def TSC_C_Thread   : TSCAttr< "tsc_c_thread", 3 >;

let cppNamespace = "::vast::hl" in
def ThreadStorage : TSCList< "TSClass", "thread storage class",
[
  TSC_None, TSC_GNU_Thread, TSC_CXX_Thread, TSC_C_Thread
] >;

class ThreadStorageClassImpl
{
  code threadStorageClassImpl = [{
    constexpr static auto thread_storage_class = "threadStorageClass";

    void setThreadStorageClass(TSClass spec) { setAttr< TSClassAttr >(thread_storage_class, spec); }
  }];
}

// Storage Duration

class StorageDurationAttr< string name, int val > : I64EnumAttrCase< name, val > {}

class StorageDurationList< string name, string summary, list<StorageDurationAttr> cases >
  : I64EnumAttr< name, summary, cases >
{}

def SD_FullExpression : StorageDurationAttr< "sd_none", 0 >;
def SD_Automatic      : StorageDurationAttr< "sd_automatic", 1 >;
def SD_Thread 	      : StorageDurationAttr< "sd_thread", 2 >;
def SD_Static         : StorageDurationAttr< "sd_static", 3 >;
def SD_Dynamic        : StorageDurationAttr< "sd_dynamic", 4 >;

let cppNamespace = "::vast::hl" in
def StorageDuration : StorageDurationList< "StorageDuration", "storage duration",
[
  SD_FullExpression, SD_Automatic, SD_Thread, SD_Static, SD_Dynamic
] >;

class StorageDurationImpl
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
class DeclContextAttr< string name, int val > : I64EnumAttrCase< name, val > {}

class DeclContextList< string name, string summary, list<DeclContextAttr> cases >
  : I64EnumAttr< name, summary, cases >
{}

def DC_Function        : DeclContextAttr< "dc_function", 0 >;
def DC_Method          : DeclContextAttr< "dc_method", 1 >;
def DC_Block           : DeclContextAttr< "dc_block", 2 >;
def DC_Capture         : DeclContextAttr< "dc_capture", 3 >;
def DC_TranslationUnit : DeclContextAttr< "dc_translation_unit", 4 >;
def DC_Record          : DeclContextAttr< "dc_record", 5 >;
def DC_Enum            : DeclContextAttr< "dc_enum", 6 >;
def DC_Namespace       : DeclContextAttr< "dc_namespace", 7 >;

let cppNamespace = "::vast::hl" in
def DeclContextKind : DeclContextList< "DeclContextKind", "declaration context kind",
[
  DC_Function, DC_Method, DC_Block, DC_Capture,
  DC_TranslationUnit, DC_Record, DC_Enum, DC_Namespace
] >;

class DeclContextImpl
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

class StorageSpecifiers :
  StorageClassImpl, ThreadStorageClassImpl, StorageDurationImpl, DeclContextImpl
{
  code storageSpecifiersImpl =
    storageClassImpl #
    threadStorageClassImpl #
    storageDurationImpl #
    declContextImpl;
}

// Variable Operation
def HighLevel_VarDeclOp : HighLevel_Op< "var", [VarSymbol] >
  , StorageSpecifiers
{
  let summary = "VAST variable declaration";
  let description = [{ VAST variable declaration }];

  let arguments = (ins
    StrAttr:$name,
    OptionalAttr<StorageClass>:$storageClass,
    OptionalAttr<ThreadStorage>:$threadStorageClass
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
      "llvm::StringRef":$name,
      CArg< "BuilderCallback", "std::nullopt" >:$initBuilder,
      CArg< "BuilderCallback", "std::nullopt" >:$allocaBuilder
    )>
  ];

  let assemblyFormat = [{
    $name attr-dict ($storageClass^)? ($threadStorageClass^)? `:` type($result)
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
