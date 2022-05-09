// Copyright (c) 2021-present, Trail of Bits, Inc.

#ifndef VAST_DIALECT_HIGHLEVEL_PASSES_TD
#define VAST_DIALECT_HIGHLEVEL_PASSES_TD

include "mlir/Pass/PassBase.td"

def LLVMDump : Pass<"vast-llvm-dump", "mlir::ModuleOp"> {
  let summary = "Pass for developers to quickly dump module as llvm ir.";
  let description = [{
    Lowers module into llvm IR and dumps it on stderr.
  }];

  let dependentDialects = ["mlir::LLVM::LLVMDialect", "vast::hl::HighLevelDialect"];
  let constructor = "vast::hl::createLLVMDumpPass()";
}

def LowerHighLevelTypes : Pass<"vast-hl-lower-types", "mlir::ModuleOp"> {
  let summary = "Lower high-level types to standard types";
  let description = [{
    Lower high-level types into standard types which is usually required first step
    by other passes in the pipeline.

    Information about bit sizes of high level types is inferred from the data layout of
    the module, which is derived from the information provided by clang and emitted
    automatically by `vast-cc`.

    TODO: Named types are not yet supported.
  }];

  let constructor = "vast::hl::createLowerHighLevelTypesPass()";
}

def LowerHLToLL : Pass<"vast-hl-lower-to-ll", "mlir::ModuleOp"> {
  let summary = "HL -> LL conversion";
  let description = [{
    Pass lowers high-level operations into low-level (for now, llvm dialect is used) dialects.
    Operations in other dialects are not touched and kept as they are. Requires types to be
    in standard dialect (only on relevant instructions).

    This pass is still a work in progress.
  }];

  let constructor = "vast::hl::createLowerHLToLLPass()";
  let dependentDialects = ["mlir::LLVM::LLVMDialect"];
}

def StructsToTuples : Pass<"vast-hl-structs-to-tuples", "mlir::ModuleOp"> {
  let summary = "Transform hl.record into std tuples.";
  let description = [{
    This pass is still a work in progress.
  }];

  let constructor = "vast::hl::createStructsToTuplesPass()";
  let dependentDialects = ["mlir::LLVM::LLVMDialect"];
}

def LowerHighLevelControlFlow : Pass<"vast-hl-lower-cf", "mlir::ModuleOp"> {
  let summary = "Lower control flow constructs into SCF.";
  let description = [{
    Pass lowers high-level control flow constructs (such as `IfOp` for example) to their
    equivalents in `SCF` dialect. Requires types on relevant operations to be in standard
    dialect.

    This pass is still a work in progress.
  }];

  let dependentDialects = ["mlir::scf::SCFDialect", "mlir::LLVM::LLVMDialect"];
  let constructor = "vast::hl::createLowerHighLevelControlFlowPass()";
}


#endif // VAST_DIALECT_HIGHLEVEL_PASSES_TD
