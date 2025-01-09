// Copyright (c) 2024, Trail of Bits, Inc.

#ifndef VAST_DIALECT_CORE_COMMON_ATTR_CONSTRAINTS_TD
#define VAST_DIALECT_CORE_COMMON_ATTR_CONSTRAINTS_TD

include "mlir/IR/CommonAttrConstraints.td"

//
// Attributes for symbol references
//

def Core_VarSymbolRefAttr : Attr<
  CPred< "::llvm::isa< ::vast::core::VarSymbolRefAttr >($_self)" >,
  "variable symbol reference attribute"
> {
  let storageType = [{ ::vast::core::VarSymbolRefAttr }];
  let returnType = [{ ::llvm::StringRef }];
  let valueType = NoneType;
  let constBuilderCall =
    "::mlir::SymbolRefAttr::get($_builder.getContext(), $0)";
  let convertFromStorage = "$_self.getValue()";
}

#endif // VAST_DIALECT_CORE_COMMON_ATTR_CONSTRAINTS_TD
