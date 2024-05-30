// Copyright (c) 2022-present, Trail of Bits, Inc.

#ifndef VAST_DIALECT_LOWLEVEL_PASSES_TD
#define VAST_DIALECT_LOWLEVEL_PASSES_TD

include "mlir/Pass/PassBase.td"

def ToLLVM : Pass<"vast-ll-to-llvm", "core::ModuleOp"> {
  let summary = "Convert low level operations to LLVM dialect.";
  let description = [{
    Work in progess.
  }];

  let dependentDialects = [
    "mlir::LLVM::LLVMDialect",
    "vast::ll::LowLevelDialect",
    "vast::core::CoreDialect"
  ];

  let constructor = "vast::ll::createToLLVMPass()";
}

#endif // VAST_DIALECT_LOWLEVEL_PASSES_TD
