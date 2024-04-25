// Copyright (c) 2022-present, Trail of Bits, Inc.

#ifndef VAST_CONVERSION_PASSES_TD
#define VAST_CONVERSION_PASSES_TD

include "mlir/Pass/PassBase.td"

#ifdef VAST_ENABLE_PDLL_CONVERSIONS

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
    "mlir::func::FuncDialect", "mlir::pdl::PDLDialect",
    "mlir::pdl_interp::PDLInterpDialect",
    "vast::core::CoreDialect"
  ];
}

#endif // VAST_ENABLE_PDLL_CONVERSIONS

def HLToLLCF : Pass<"vast-hl-to-ll-cf", "mlir::ModuleOp"> {
  let summary = "VAST HL control flow to LL control flow";
  let description = [{
    Transforms high level control flow operations into their low level
    representation.

    This pass is still a work in progress.
  }];

  let constructor = "vast::createHLToLLCFPass()";
  let dependentDialects = [
    "vast::ll::LowLevelDialect",
    "vast::core::CoreDialect"
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
  let dependentDialects = [
    "mlir::LLVM::LLVMDialect",
    "vast::core::CoreDialect",
    "vast::ll::LowLevelDialect",
    "vast::hl::HighLevelDialect",
  ];
}

def FnArgsToAlloca : Pass<"vast-fn-args-to-alloca", "mlir::ModuleOp"> {
  let summary = "VAST to LLVM Dialect conversion";
  let description = [{
    For each function emit its prologue - for each argument make an alloca
    and store the corresponding argument in it.
  }];

  let constructor = "vast::createFnArgsToAllocaPass()";
  let dependentDialects = [
    "vast::ll::LowLevelDialect",
    "vast::core::CoreDialect"
  ];
}

def LowerValueCategories : Pass<"vast-lower-value-categories", "mlir::ModuleOp"> {
  let summary = "Lower `hl.lvalue` into explicit pointers and loads.";
  let description = [{
    Lower `hl.lvalue` into explicit memory. This changes types to pointers and emits
    explicit load operations.
  }];

  let constructor = "vast::createLowerValueCategoriesPass()";
  let dependentDialects = [
    "vast::ll::LowLevelDialect",
    "vast::hl::HighLevelDialect",
  ];
}

def CoreToLLVM : Pass<"vast-core-to-llvm", "mlir::ModuleOp"> {
  let summary = "VAST Core dialect to LLVM Dialect conversion";
  let description = [{
    Converts core dialect operations to LLVM dialect.
    }];

  let constructor = "vast::createCoreToLLVMPass()";
  let dependentDialects = [
    "mlir::LLVM::LLVMDialect",
    "vast::core::CoreDialect"
  ];
}

def EmitABI : Pass<"vast-emit-abi", "mlir::ModuleOp"> {
  let summary = "Transform functions and apply abi conversion to their type.";
  let description = [{
    This pass is still a work in progress.
  }];

  let constructor = "vast::createEmitABIPass()";
  let dependentDialects = [
    "vast::abi::ABIDialect",
    "vast::core::CoreDialect"
  ];
}

def LowerABI : Pass<"vast-lower-abi", "mlir::ModuleOp"> {
  let summary = "Lower abi operations.";
  let description = [{
    This pass is still a work in progress.
  }];

  let constructor = "vast::createLowerABIPass()";
  let dependentDialects = [
    "mlir::LLVM::LLVMDialect",
    "vast::ll::LowLevelDialect",
    "vast::core::CoreDialect"
  ];
}

def HLToLLGEPs : Pass<"vast-hl-to-ll-geps", "mlir::ModuleOp"> {
  let summary = "Convert hl.member to ll.gep";
  let description = [{
    This pass is still a work in progress.
  }];

  let constructor = "vast::createHLToLLGEPsPass()";
  let dependentDialects = [
    "vast::ll::LowLevelDialect",
    "vast::core::CoreDialect"
  ];
}

def HLToLLVars : Pass<"vast-hl-to-ll-vars", "mlir::ModuleOp"> {
  let summary = "Convert hl variables into ll versions.";
  let description = [{
    This pass is still a work in progress.
  }];

  let constructor = "vast::createHLToLLVarsPass()";
  let dependentDialects = [
    "mlir::LLVM::LLVMDialect",
    "vast::ll::LowLevelDialect",
    "vast::core::CoreDialect"
  ];
}

def HLToLLFunc : Pass<"vast-hl-to-ll-func", "mlir::ModuleOp"> {
  let summary = "Convert hl functions into ll versions.";
  let description = [{
    This pass is still a work in progress.
  }];

  let constructor = "vast::createHLToLLFuncPass()";
  let dependentDialects = [
    "vast::ll::LowLevelDialect",
    "vast::core::CoreDialect"
  ];
}

def HLEmitLazyRegions : Pass<"vast-hl-to-lazy-regions", "mlir::ModuleOp"> {
  let summary = "Transform hl operations that have short-circuiting into lazy operations.";
  let description = [{
    This pass is still a work in progress.
  }];

  let constructor = "vast::createHLEmitLazyRegionsPass()";
  let dependentDialects = [
    "vast::core::CoreDialect"
  ];
}

def HLToHLBI : Pass<"vast-hl-to-hl-builtin", "mlir::ModuleOp"> {
  let summary = "Transform hl operations that have builtin attribute to specialized dialect.";
  let description = [{
    This pass is still a work in progress.
  }];

  let constructor = "vast::createHLToHLBI()";
  let dependentDialects = [
    "vast::hlbi::HLBuiltinDialect"
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

  let constructor = "vast::createExportFnInfoPass()";

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

  let constructor = "vast::createDCEPass()";
}

def UDE : Pass<"vast-hl-ude", "mlir::ModuleOp"> {
  let summary = "Eliminate unused definitions";
  let description = [{
    Removes unused definitions, such as typedefs, structs and functions.
  }];

  let dependentDialects = [
    "vast::hl::HighLevelDialect",
    "vast::core::CoreDialect"
  ];

  let constructor = "vast::createUDEPass()";
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

  let constructor = "vast::createHLLowerTypesPass()";
}

def LowerTypeDefs : Pass<"vast-hl-lower-typedefs", "mlir::ModuleOp"> {
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

  let constructor = "vast::createLowerTypeDefsPass()";
}

def LowerElaboratedTypes : Pass<"vast-hl-lower-elaborated-types", "mlir::ModuleOp"> {
  let summary = "Replace `hl::ElaboratedType` type by its underlying type.";
  let description = [{
    Replaces `hl::ElaboratedType` types by its underlying type.

    All `hl::ElaboratedType` are marked illegal and converted by this pass.
  }];

  let dependentDialects = [
    "vast::hl::HighLevelDialect",
    "vast::core::CoreDialect"
  ];

  let constructor = "vast::createLowerElaboratedTypesPass()";
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

  let constructor = "vast::createSpliceTrailingScopes()";
}

#endif // VAST_CONVERSION_PASSES_TD
