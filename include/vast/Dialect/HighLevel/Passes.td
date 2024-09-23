// Copyright (c) 2021-present, Trail of Bits, Inc.

#ifndef VAST_DIALECT_HIGHLEVEL_PASSES_TD
#define VAST_DIALECT_HIGHLEVEL_PASSES_TD

include "mlir/Pass/PassBase.td"

def ExportFnInfo : Pass<"vast-export-fn-info", "core::ModuleOp"> {
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

def DCE : Pass<"vast-hl-dce", "core::ModuleOp"> {
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

def UDE : Pass<"vast-hl-ude", "core::ModuleOp"> {
  let summary = "Eliminate unused definitions";
  let description = [{
    Removes unused definitions, such as typedefs, structs and functions.
  }];

  let dependentDialects = [
    "vast::hl::HighLevelDialect",
    "vast::core::CoreDialect"
  ];

  let constructor = "vast::hl::createUDEPass()";
}

def HLLowerTypes : Pass<"vast-hl-lower-types", "core::ModuleOp"> {
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

def LowerTypeDefs : Pass<"vast-hl-lower-typedefs", "core::ModuleOp"> {
  let summary = "Replace `hl::TypeDef` type by its underlying aliased type.";
  let description = [{
    Replaces `hl::TypeDef` types by its underlying aliased types.
    The conversion resolves nested typedefs.

    All `hl::TypeDef` are marked illegal and converted by this pass.
  }];

  let dependentDialects = [
    "vast::hl::HighLevelDialect",
    "vast::core::CoreDialect"
  ];

  let constructor = "vast::hl::createLowerTypeDefsPass()";
}

def LowerElaboratedTypes : Pass<"vast-hl-lower-elaborated-types", "core::ModuleOp"> {
  let summary = "Replace `hl::ElaboratedType` type by its underlying type.";
  let description = [{
    Replaces `hl::ElaboratedType` types by its underlying type.

    All `hl::ElaboratedType` are marked illegal and converted by this pass.
  }];

  let dependentDialects = [
    "vast::hl::HighLevelDialect",
    "vast::core::CoreDialect"
  ];

  let constructor = "vast::hl::createLowerElaboratedTypesPass()";
}

def SpliceTrailingScopes : Pass<"vast-hl-splice-trailing-scopes", "core::ModuleOp"> {
  let summary = "Remove trailing `core::ScopeOp`s.";
  let description = [{
    Removes trailing scopes.
  }];

  let dependentDialects = [
    "vast::hl::HighLevelDialect",
    "vast::core::CoreDialect"
  ];

  let constructor = "vast::hl::createSpliceTrailingScopes()";
}

def LowerEnumRefs : Pass<"vast-hl-lower-enum-refs", "core::ModuleOp"> {
  let summary = "Lower high level representation of enums.";
  let description = [{
    Lower enum references to integer values.
  }];

  let constructor = "vast::hl::createLowerEnumRefsPass()";
  let dependentDialects = [
    "vast::hl::HighLevelDialect"
  ];
}

def LowerEnumDecls : Pass<"vast-hl-lower-enum-decls", "core::ModuleOp"> {
  let summary = "Lower high level representation of enums.";
  let description = [{
    Lower enum types to integer types.

    Note: This pass requires that all enum references are lowered.
  }];

  let constructor = "vast::hl::createLowerEnumDeclsPass()";
  let dependentDialects = [
    "vast::hl::HighLevelDialect"
  ];
}

#endif // VAST_DIALECT_HIGHLEVEL_PASSES_TD
