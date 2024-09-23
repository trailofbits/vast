// Copyright (c) 2023-present, Trail of Bits, Inc.

#ifndef VAST_DIALECT_CORE_OPS
#define VAST_DIALECT_CORE_OPS

include "mlir/IR/OpBase.td"

include "mlir/Dialect/DLTI/DLTIBase.td"
include "mlir/Interfaces/DataLayoutInterfaces.td"

include "vast/Dialect/Core/CoreTraits.td"
include "vast/Dialect/Core/Interfaces/SymbolTableInterface.td"

def Core_ModuleOp : Core_Op< "module", [
  IsolatedFromAbove,
  DataLayoutOpInterface,
  HasDefaultDLTIDataLayout,
  NoTerminator, NoRegionArguments, SingleBlock, Core_Symbol,
  Core_ShadowingSymbolTable< [
    [Core_FuncSymbol, Core_VarSymbol, Core_TypeSymbol, Core_EnumConstantSymbol],
    [Core_ElaboratedTypeSymbol]
  ] >,
] > {
  let summary = "VAST top level container operation";
  let description = [{
    A `module` represents a top-level container operation. We provide custom
    module operation to allow for custom symbol table management.
  }];

  let arguments = (ins
    OptionalAttr<SymbolNameAttr>:$sym_name
  );

  let regions = (region SizedRegion<1>:$body);

  let builders = [OpBuilder<(ins CArg<"std::optional<string_ref>", "{}">:$name)>];

  let extraClassDeclaration = [{
    /// Construct a module from the given location with an optional name.
    static ModuleOp create(mlir::Location loc, std::optional<llvm::StringRef> name = std::nullopt);

    /// Return the name of this module if present.
    std::optional<llvm::StringRef> getName() { return getSymName(); }

    //===------------------------------------------------------------------===//
    // SymbolOpInterface Methods
    //===------------------------------------------------------------------===//

    /// A ModuleOp may optionally define a symbol.
    bool isOptionalSymbol() { return true; }

    //===------------------------------------------------------------------===//
    // DataLayoutOpInterface Methods
    //===------------------------------------------------------------------===//

    mlir::DataLayoutSpecInterface getDataLayoutSpec();

    //===------------------------------------------------------------------===//
    // OpAsmOpInterface Methods
    //===------------------------------------------------------------------===//

    static ::llvm::StringRef getDefaultDialect() {
      return "core";
    }
  }];

  let assemblyFormat = "($sym_name^)? attr-dict-with-keyword $body";

  let hasVerifier = 1;

  // We need to ensure the block inside the region is properly terminated;
  // the auto-generated builders do not guarantee that.
  let skipDefaultBuilders = 1;
}

def Core_ScopeOp : Core_Op< "scope", [
  NoTerminator, Core_ScopeLikeTrait,
  Core_ShadowingSymbolTable< [
    [Core_VarSymbol, Core_TypeSymbol, Core_EnumConstantSymbol],
    [Core_ElaboratedTypeSymbol]
  ] >
] > {
  let summary = "VAST scope declaration";
  let description = [{
    Scope operation serves to represent explicitly high-level code scope.
    Other control flow operations represent scopes implicitly.  It is a
    single-region operation.
  }];

  let regions = (region AnyRegion:$body);

  let assemblyFormat = [{ $body attr-dict }];
}

def Core_ImplicitReturnOp
  : Core_Op< "implicit.return", [Core_SoftTerminatorTrait, Core_ReturnLikeTrait] >
  , Arguments<(ins Variadic<AnyType>:$result)>
{
  let summary = "VAST implicit return";
  let description = [{
    Op representing return that wasn't explicitely written in the source (e.g. in void fun(){}).
  }];

  let assemblyFormat = "$result `:` type($result) attr-dict";
}

#endif //VAST_DIALECT_CORE_OPS
