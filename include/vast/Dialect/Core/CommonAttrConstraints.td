// Copyright (c) 2024, Trail of Bits, Inc.

#ifndef VAST_DIALECT_CORE_COMMON_ATTR_CONSTRAINTS_TD
#define VAST_DIALECT_CORE_COMMON_ATTR_CONSTRAINTS_TD

include "mlir/IR/CommonAttrConstraints.td"

def FuncSymbolRefAttr : Attr<
    CPred<"::llvm::isa<::vast::core::FuncSymbolRefAttr>($_self)">,
    "function symbol reference attribute"
> {
  let storageType = [{ ::vast::core::FuncSymbolRefAttr }];
  let returnType = [{ ::llvm::StringRef }];
  let valueType = NoneType;
  let constBuilderCall = [{
    ::mlir::SymbolRefAttr::get($_builder.getContext(), $0)
  }];
  let convertFromStorage = "$_self.getSymbolRef().getValue()";
}

#endif // VAST_DIALECT_CORE_COMMON_ATTR_CONSTRAINTS_TD