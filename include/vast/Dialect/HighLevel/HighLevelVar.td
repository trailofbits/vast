// Copyright (c) 2021-present, Trail of Bits, Inc.

#ifndef VAST_DIALECT_HIGHLEVEL_IR_HIGHLEVELVAR
#define VAST_DIALECT_HIGHLEVEL_IR_HIGHLEVELVAR

class StorageClassAttr< string name, int val >
  : I64EnumAttrCase< name, val >
{}

class StorageClassList< string name, string summary, list<StorageClassAttr> cases >
  : I64EnumAttr< name, summary, cases >
{}

def None          : StorageClassAttr< "sc_none", 0 >;
def Auto          : StorageClassAttr< "sc_auto", 1 >;
def Static        : StorageClassAttr< "sc_static", 2 >;
def Extern        : StorageClassAttr< "sc_extern", 3 >;
def PrivateExtern : StorageClassAttr< "sc_private_extern", 4 >;
def Register      : StorageClassAttr< "sc_register", 5 >;

let cppNamespace = "::vast::hl" in
def StorageClass : StorageClassList< "StorageClass", "storage class", [
  None, Auto, Static, Extern, PrivateExtern, Register
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

// Linkage Queries

// Storage Duration Queries

// Variable Operation
def HighLevel_VarDecl : HighLevel_Op< "var" >
  , StorageClasses
{
  let summary = "VAST variable declaration";
  let description = [{ VAST variable declaration }];

  let arguments = (ins
    StrAttr:$name,
    OptionalAttr<StorageClass>:$storageClass
  );

  let results = (outs AnyType:$result);
  let regions = (region AnyRegion:$initializer);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "Type":$type, "llvm::StringRef":$name, "BuilderCallback":$initBuilder)>
  ];

  let assemblyFormat = [{
    $name attr-dict ($storageClass^)? `:` type($result) (`=` $initializer^)?
  }];

  let extraClassDeclaration = storageClasses;
}

#endif // VAST_DIALECT_HIGHLEVEL_IR_HIGHLEVELOPS