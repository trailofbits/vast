// Copyright (c) 2021-present, Trail of Bits, Inc.

#ifndef VAST_DIALECT_HIGHLEVEL_PASSES_TD
#define VAST_DIALECT_HIGHLEVEL_PASSES_TD

include "mlir/Pass/PassBase.td"

def LLVMDump : Pass<"vast-llvm-dump", "mlir::ModuleOp"> {
  let summary = "Pass for developers to quickly dump module as llvm ir.";
  let description = [{
    Lowers module into llvm IR and dumps it on stderr.
  }];

  let dependentDialects = [
    "mlir::LLVM::LLVMDialect",
    "vast::hl::HighLevelDialect",
    "vast::core::CoreDialect"
  ];

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

  let dependentDialects = [
    "vast::hl::HighLevelDialect",
    "vast::core::CoreDialect"
  ];

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

  let dependentDialects = [
    "vast::hl::HighLevelDialect",
    "vast::core::CoreDialect"
  ];

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

  let dependentDialects = [
    "vast::hl::HighLevelDialect",
    "vast::core::CoreDialect"
  ];

  let constructor = "vast::hl::createHLLowerTypesPass()";
}

def ResolveTypeDefs : Pass<"vast-hl-resolve-typedefs", "mlir::ModuleOp"> {
  let summary = "Remove `hl::TypeDef type from types.";
  let description = [{
    Resolve typedefs types - this is usefull for more low-level analysis
    or transformations.
  }];

  let dependentDialects = [
    "vast::hl::HighLevelDialect",
    "vast::core::CoreDialect"
  ];

  let constructor = "vast::hl::createResolveTypeDefsPass()";
}

def SpliceTrailingScopes : Pass<"vast-hl-splice-trailing-scopes", "mlir::ModuleOp"> {
  let summary = "Remove trailing `hl::Scope`s.";
  let description = [{
    Removes trailing scopes.
  }];

  let dependentDialects = [
    "vast::hl::HighLevelDialect",
    "vast::core::CoreDialect"
  ];

  let constructor = "vast::hl::createSpliceTrailingScopes()";
}

#endif // VAST_DIALECT_HIGHLEVEL_PASSES_TD
