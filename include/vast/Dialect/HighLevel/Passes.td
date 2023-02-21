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

  let options = [
    Option< "bitcode_file", "bc-file", "std::string", "",
            "Specify file where to dump the bitcode" >
  ];
}

def ExportFnInfo : Pass<"vast-export-fn-info", "mlir::ModuleOp"> {
  let summary = "Create JSON that exports information about function arguments.";
  let description = [{
    Lowers module into llvm IR and dumps it on stderr.
  }];

  let dependentDialects = ["vast::hl::HighLevelDialect"];
  let constructor = "vast::hl::createExportFnInfoPass()";

  let options = [
    Option< "o", "o", "std::string", "",
            "Output JSON file to be created." >
  ];
}

def DCE : Pass<"vast-hl-dce", "mlir::ModuleOp"> {
  let summary = "Trim dead code";
  let description = [{
    Removes unreachable code, such as code after return or break/continue.
  }];

  let dependentDialects = ["vast::hl::HighLevelDialect"];
  let constructor = "vast::hl::createDCEPass()";
}

def HLLowerTypes : Pass<"vast-hl-lower-types", "mlir::ModuleOp"> {
  let summary = "Lower high-level types to standard types";
  let description = [{
    Lower high-level types into standard types which is usually required first step
    by other passes in the pipeline.

    Information about bit sizes of high level types is inferred from the data layout of
    the module, which is derived from the information provided by clang and emitted
    automatically by `vast-cc`.

    TODO: Named types are not yet supported.
  }];

  let constructor = "vast::hl::createHLLowerTypesPass()";
}

def HLLowerEnums : Pass<"vast-hl-lower-enums", "mlir::ModuleOp"> {
  let summary = "Lower high-level enums and their usages to their underlying types.";
  let description = [{
    Lower enum usages to their underlying types - this will effectively remove the enum itself.
  }];

  let constructor = "vast::hl::createHLLowerEnumsPass()";
}

def HLToLLGEPs : Pass<"vast-hl-to-ll-geps", "mlir::ModuleOp"> {
  let summary = "Convert hl.member to ll.gep";
  let description = [{
    This pass is still a work in progress.
  }];

  let constructor = "vast::hl::createHLToLLGEPsPass()";
  let dependentDialects = ["mlir::LLVM::LLVMDialect", "vast::ll::LowLevelDialect"];
}

def HLToLLVars : Pass<"vast-hl-to-ll-vars", "mlir::ModuleOp"> {
  let summary = "Convert hl variables into ll versions.";
  let description = [{
    This pass is still a work in progress.
  }];

  let constructor = "vast::hl::createHLToLLVarsPass()";
  let dependentDialects = ["mlir::LLVM::LLVMDialect", "vast::ll::LowLevelDialect"];
}

def HLStructsToTuples : Pass<"vast-hl-structs-to-tuples", "mlir::ModuleOp"> {
  let summary = "Transform hl.struct into std tuples.";
  let description = [{
    This pass is still a work in progress.
  }];

  let constructor = "vast::hl::createHLStructsToTuplesPass()";
  let dependentDialects = ["mlir::LLVM::LLVMDialect"];
}


def HLStructsToLLVM : Pass<"vast-hl-structs-to-llvm", "mlir::ModuleOp"> {
  let summary = "Transform hl.struct into llvm types without applying ABI conversions.";
  let description = [{
    This pass is still a work in progress.
  }];

  let constructor = "vast::hl::createHLStructsToLLVMPass()";
  let dependentDialects = ["mlir::LLVM::LLVMDialect"];
}

def HLToSCF : Pass<"vast-hl-to-scf", "mlir::ModuleOp"> {
  let summary = "Lower control flow constructs into SCF.";
  let description = [{
    Pass lowers high-level control flow constructs (such as `IfOp` for example) to their
    equivalents in `SCF` dialect. Requires types on relevant operations to be in standard
    dialect.

    This pass is still a work in progress.
  }];

  let dependentDialects = ["mlir::scf::SCFDialect", "mlir::LLVM::LLVMDialect"];
  let constructor = "vast::hl::createHLToSCFPass()";
}

def HLEmitLazyRegions : Pass<"vast-hl-to-lazy-regions", "mlir::ModuleOp"> {
  let summary = "Transform hl operations that have short-circuiting into lazy operations.";
  let description = [{
    This pass is still a work in progress.
  }];

  let constructor = "vast::hl::createHLEmitLazyRegionsPass()";
  let dependentDialects = ["vast::core::CoreDialect"];
}

#endif // VAST_DIALECT_HIGHLEVEL_PASSES_TD
