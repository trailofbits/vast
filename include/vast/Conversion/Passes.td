// Copyright (c) 2022-present, Trail of Bits, Inc.

#ifndef VAST_CONVERSION_PASSES_TD
#define VAST_CONVERSION_PASSES_TD

include "mlir/Pass/PassBase.td"

#ifdef ENABLE_PDLL_CONVERSIONS

def HLToFunc : Pass<"vast-hl-to-func", "mlir::ModuleOp"> {
  let summary = "Lower high-level functions to func.func operations.";
  let description = [{
    Lowers high-level function operations to function dialect.

    Drops high-level information like linkage attributes.
  }];

  let constructor = "vast::createHLToFuncPass()";
  let dependentDialects = [
    "vast::hl::HighLevelDialect", "mlir::func::FuncDialect",
    "mlir::pdl::PDLDialect", "mlir::pdl_interp::PDLInterpDialect"
  ];
}

#endif // ENABLE_PDLL_CONVERSIONS

def HLFuncToFunc : Pass<"vast-hl-to-func", "mlir::ModuleOp"> {
  let summary = "Lower high-level functions to func.func operations.";
  let description = [{
    Lowers high-level function operations to function dialect.

    Drops high-level information like linkage attributes.
  }];

  let constructor = "vast::createHLFuncToFuncPass()";
  let dependentDialects = [
    "vast::hl::HighLevelDialect", "mlir::func::FuncDialect"
  ];
}

def CoreToLLVM : Pass<"vast-core-to-llvm", "mlir::ModuleOp"> {
  let summary = "VAST to LLVM Dialect conversion";
  let description = [{
    Converts lowest level VAST operations to LLVM dialect. It is expected
    that module being converted was already lowered by other VAST passes.

    This pass is still a work in progress.
  }];

  let constructor = "vast::createCoreToLLVMPass()";
  let dependentDialects = ["mlir::LLVM::LLVMDialect", "vast::ll::LowLevelDialect",
                           "vast::hl::HighLevelDialect"];
}

#endif // VAST_CONVERSION_PASSES_TD
