// Copyright (c) 2023, Trail of Bits, Inc.

#ifndef VAST_DIALECT_CORE_FUNC
#define VAST_DIALECT_CORE_FUNC

include "mlir/IR/OpBase.td"
include "mlir/IR/FunctionInterfaces.td"

include "mlir/Interfaces/CallInterfaces.td"
include "mlir/IR/RegionKindInterface.td"

include "vast/Interfaces/SymbolInterface.td"

def GlobalLinkageKind : Attr<
    CPred<"::llvm::isa<::vast::core::GlobalLinkageKindAttr>($_self)">,
    "global linkage kind"
> {
    let storageType = "::vast::core::GlobalLinkageKindAttr";
    let returnType  = "::vast::core::GlobalLinkageKind";
    let convertFromStorage = "$_self.getValue()";
    let constBuilderCall = [{
        ::vast::core::GlobalLinkageKindAttr::get($_builder.getContext(), $0)
    }];
}

def CoreFunctionType : Type<
    CPred<"::llvm::isa<::vast::core::FunctionType>($_self)">,
    "function type",
    "::vast::core::FunctionType"
>;

class Core_FuncBaseOp< Dialect dialect, string mnemonic, list< Trait > traits = [] >
  : Op< dialect, mnemonic,
    !listconcat(traits, [
      AutomaticAllocationScope,
      CallableOpInterface,
      FunctionOpInterface,
      IsolatedFromAbove,
      Symbol,
      NoTerminator,
      DeclareOpInterfaceMethods< RegionKindInterface >
    ])
> {
  let summary = "VAST function template";

  let description = [{
    Inspired by `cir::FuncOp` and `mlir::func::FuncOp`:
    > Operations within the function cannot implicitly capture values defined
    > outside of the function, i.e. Functions are `IsolatedFromAbove`. All
    > external references must use function arguments or attributes that establish
    > a symbolic connection (e.g. symbols referenced by name via a string
    > attribute like SymbolRefAttr). An external function declaration (used when
    > referring to a function declared in some other module) has no body.

    The function linkage information is specified by `linkage`, as defined by
    `GlobalLinkageKind` attribute.
  }];

  let arguments =(ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<CoreFunctionType>:$function_type,
    DefaultValuedAttr<GlobalLinkageKind, "core::GlobalLinkageKind::ExternalLinkage">:$linkage,
    OptionalAttr<StrAttr>:$sym_visibility,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs
  );

  let regions = (region AnyRegion:$body);

  let extraClassDeclaration = [{
    bool isVarArg() { return getFunctionType().isVarArg(); }

    //===------------------------------------------------------------------===//
    // CallableOpInterface
    //===------------------------------------------------------------------===//

    /// Returns the region on the current operation that is callable. This may
    /// return null in the case of an external callable object, e.g. an external
    /// function.
    ::mlir::Region *getCallableRegion() {
      return isExternal() ? nullptr : &getBody();
    }

    /// Returns the results types that the callable region produces when
    /// executed.
    llvm::ArrayRef<Type> getCallableResults() {
      return getFunctionType().getResults();
    }

    /// Returns the argument attributes for all callable region arguments or
    /// null if there are none.
    ::mlir::ArrayAttr getCallableArgAttrs() {
      return getArgAttrs().value_or(nullptr);
    }

    /// Returns the result attributes for all callable region results or
    /// null if there are none.
    ::mlir::ArrayAttr getCallableResAttrs() {
      return getResAttrs().value_or(nullptr);
    }

    //===------------------------------------------------------------------===//
    // FunctionOpInterface Methods
    //===------------------------------------------------------------------===//

    /// Returns the argument types of this function.
    llvm::ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this function.
    llvm::ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }

    //===------------------------------------------------------------------===//
    // SymbolOpInterface Methods
    //===------------------------------------------------------------------===//

    bool isDeclaration() { return isExternal(); }
  }];
}

#endif // VAST_DIALECT_CORE_FUNC
