// Copyright (c) 2022-present, Trail of Bits, Inc.

#ifndef VAST_CONVERSION_PASSES_TD
#define VAST_CONVERSION_PASSES_TD

include "mlir/Pass/PassBase.td"

#ifdef ENABLE_PDLL_CONVERSIONS

// TODO(lukas): Figure out better naming for passes that are implemented in both PDLL and C++
//              (if it becomes more common occurence).
def HLToFunc_pdll : Pass<"vast-hl-to-func-pdll", "mlir::ModuleOp"> {
  let summary = "Lower high-level functions to func.func operations.";
  let description = [{
    Lowers high-level function operations to function dialect.
    Drops high-level information like linkage attributes.

    Work in progress (currently does not work properly).
  }];

  let cppnamespace = pdll;

  let constructor = "vast::pdll::createHLToFuncPass()";
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

def IRsToLLVM : Pass<"vast-irs-to-llvm", "mlir::ModuleOp"> {
  let summary = "VAST to LLVM Dialect conversion";
  let description = [{
    Converts lowest level VAST operations to LLVM dialect. It is expected
    that module being converted was already lowered by other VAST passes.

    This pass is still a work in progress.
  }];

  let constructor = "vast::createIRsToLLVMPass()";
  let dependentDialects = ["mlir::LLVM::LLVMDialect", "vast::ll::LowLevelDialect",
                           "vast::hl::HighLevelDialect"];
}

def CoreToLLVM : Pass<"vast-core-to-llvm", "mlir::ModuleOp"> {
  let summary = "VAST Core dialec to LLVM Dialect conversion";
  let description = [{
    Converts core dialect operations to LLVM dialect.
    }];

  let constructor = "vast::createCoreToLLVMPass()";
  let dependentDialects = ["mlir::LLVM::LLVMDialect", "vast::core::CoreDialect",
                           "vast::hl::HighLevelDialect"];
}

#endif // VAST_CONVERSION_PASSES_TD
