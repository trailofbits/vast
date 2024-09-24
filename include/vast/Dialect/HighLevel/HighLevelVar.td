// Copyright (c) 2021-present, Trail of Bits, Inc.

#ifndef VAST_DIALECT_HIGHLEVEL_IR_HIGHLEVELVAR
#define VAST_DIALECT_HIGHLEVEL_IR_HIGHLEVELVAR

include "vast/Dialect/Core/CoreTraits.td"
include "vast/Dialect/Core/StorageInfo.td"
include "vast/Dialect/Core/Interfaces/DeclStorageInterface.td"

// Variable Operation
def HighLevel_VarDeclOp
  : HighLevel_Op< "var", [
      Core_VarSymbol,
      Core_DeclStorageInterface,
      DeclareOpInterfaceMethods< Core_DeclStorageInterface >
  ] >
{
  let summary = "VAST variable declaration";
  let description = [{ VAST variable declaration }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttr:$type,
    Core_StorageClass:$storageClass,
    Core_ThreadStorage:$threadStorageClass
  );

  let regions = (region
    AnyRegion:$initializer,
    AnyRegion:$allocation_size
  );

  let builders = [
    OpBuilder<(ins
      "Type":$type,
      "llvm::StringRef":$sym_name,
      "core::StorageClass":$storageClass,
      "core::TSClass":$threadStorageClass,
      CArg< "maybe_builder_callback_ref", "std::nullopt" >:$initBuilder,
      CArg< "maybe_builder_callback_ref", "std::nullopt" >:$allocaBuilder
    )>
  ];

  let assemblyFormat = [{
    $sym_name attr-dict custom< StorageClasses >($storageClass, $threadStorageClass)
      `:` $type
      (`=` $initializer^)?
      (`allocation_size` $allocation_size^)?
  }];
}

#endif // VAST_DIALECT_HIGHLEVEL_IR_HIGHLEVELVAR
