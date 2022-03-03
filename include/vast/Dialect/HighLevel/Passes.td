// Copyright (c) 2021-present, Trail of Bits, Inc.

#ifndef VAST_DIALECT_HIGHLEVEL_PASSES_TD
#define VAST_DIALECT_HIGHLEVEL_PASSES_TD

include "mlir/Pass/PassBase.td"

def LowerHighLevelTypes : Pass<"vast-hl-lower-types"> {
  let summary = "Lower high-level types to standard types";
  let description = [{ }];

  let constructor = "vast::hl::createLowerHighLevelTypesPass()";
}


def StructsToTuples : Pass<"vast-hl-structs-to-tuples", "mlir::ModuleOp"> {
  let summary = "Transform hl.record into std tuples.";
  let description = [{
    This pass is still a work in progress.
  }];

  let constructor = "vast::hl::createStructsToTuplesPass()";
  let dependentDialects = ["mlir::LLVM::LLVMDialect"];
}

def LowerHighLevelControlFlow : Pass<"vast-hl-lower-cf"> {
  let summary = "Lower high-level types to standard types";
  let description = [{ }];

  let constructor = "vast::hl::createLowerHighLevelControlFlowPass()";
}


#endif // VAST_DIALECT_HIGHLEVEL_PASSES_TD
