// Copyright (c) 2021-present, Trail of Bits, Inc.

#ifndef VAST_DIALECT_HIGHLEVEL_IR_HIGHLEVELVAR
#define VAST_DIALECT_HIGHLEVEL_IR_HIGHLEVELVAR

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

class StorageClasses
{
    code storageClasses = [{
        constexpr static auto storage_class = "storageClass";

        StorageClass getStorageClass() {
            return storageClassAttr().getValue();
        }

        void setStorageClass(StorageClass spec) {
            auto ctx = this->getContext();
            (*this)->setAttr(storage_class, StorageClassAttr::get(ctx, spec));
        }
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

class ThreadStorageClasses
{
    code threadStorageClasses = [{
        constexpr static auto thread_storage_class = "threadStorageClass";

        TSClass getThreadStorageClass() {
            return threadStorageClassAttr().getValue();
        }

        void setThreadStorageClass(TSClass spec) {
            auto ctx = this->getContext();
            (*this)->setAttr(thread_storage_class, TSClassAttr::get(ctx, spec));
        }
    }];
}

// Storage Duration

class StorageDurationAttr< string name, int val > : I64EnumAttrCase< name, val > {}

class StorageDurationList< string name, string summary, list<StorageDurationAttr> cases >
  : I64EnumAttr< name, summary, cases >
{}

def SD_FullExpression : StorageDurationAttr< "sd_none", 0 >;
def SD_Automatic      : StorageDurationAttr< "sd_automatic", 0 >;
def SD_Thread 	      : StorageDurationAttr< "sd_thread", 0 >;
def SD_Static         : StorageDurationAttr< "sd_static", 0 >;
def SD_Dynamic        : StorageDurationAttr< "sd_dynamic", 1 >;

let cppNamespace = "::vast::hl" in
def StorageDuration : StorageDurationList< "StorageDuration", "storage duration",
[
  SD_FullExpression, SD_Automatic, SD_Thread, SD_Static, SD_Dynamic
] >;

// Variable Operation
def HighLevel_VarDecl : HighLevel_Op< "var", [Symbol] >
  , StorageClasses, ThreadStorageClasses
{
  let summary = "VAST variable declaration";
  let description = [{ VAST variable declaration }];

  let arguments = (ins
    TypeAttr:$type,
    SymbolNameAttr:$sym_name,
    OptionalAttr<StorageClass>:$storageClass,
    OptionalAttr<ThreadStorage>:$threadStorageClass
  );

  let regions = (region AnyRegion:$initializer);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "Type":$type, "llvm::StringRef":$name, "BuilderCallback":$initBuilder)>
  ];

  let assemblyFormat = [{
    $sym_name ($storageClass^)? ($threadStorageClass^)? attr-dict `:` $type (`=` $initializer^)?
  }];

  let extraClassDeclaration = storageClasses # threadStorageClasses;
}

#endif // VAST_DIALECT_HIGHLEVEL_IR_HIGHLEVELOPS
