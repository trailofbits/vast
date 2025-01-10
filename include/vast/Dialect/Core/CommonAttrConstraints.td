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

def Core_TypeSymbolRefAttr : Attr<
  CPred< "::llvm::isa< ::vast::core::TypeSymbolRefAttr >($_self)" >,
  "type symbol reference attribute"
> {
  let storageType = [{ ::vast::core::TypeSymbolRefAttr }];
  let returnType = [{ ::llvm::StringRef }];
  let valueType = NoneType;
  let constBuilderCall =
    "::mlir::SymbolRefAttr::get($_builder.getContext(), $0)";
  let convertFromStorage = "$_self.getValue()";
}

def Core_FuncSymbolRefAttr : Attr<
  CPred< "::llvm::isa< ::vast::core::FuncSymbolRefAttr >($_self)" >,
  "function symbol reference attribute"
> {
  let storageType = [{ ::vast::core::FuncSymbolRefAttr }];
  let returnType = [{ ::llvm::StringRef }];
  let valueType = NoneType;
  let constBuilderCall =
    "::mlir::SymbolRefAttr::get($_builder.getContext(), $0)";
  let convertFromStorage = "$_self.getValue()";
}

def Core_LabelSymbolRefAttr : Attr<
  CPred< "::llvm::isa< ::vast::core::LabelSymbolRefAttr >($_self)" >,
  "label symbol reference attribute"
> {
  let storageType = [{ ::vast::core::LabelSymbolRefAttr }];
  let returnType = [{ ::llvm::StringRef }];
  let valueType = NoneType;
  let constBuilderCall =
    "::mlir::SymbolRefAttr::get($_builder.getContext(), $0)";
  let convertFromStorage = "$_self.getValue()";
}

def Core_EnumConstantSymbolRefAttr : Attr<
  CPred< "::llvm::isa< ::vast::core::EnumConstantSymbolRefAttr >($_self)" >,
  "enum constant symbol reference attribute"
> {
  let storageType = [{ ::vast::core::EnumConstantSymbolRefAttr }];
  let returnType = [{ ::llvm::StringRef }];
  let valueType = NoneType;
  let constBuilderCall =
    "::mlir::SymbolRefAttr::get($_builder.getContext(), $0)";
  let convertFromStorage = "$_self.getValue()";
}

def Core_MemberVarSymbolRefAttr : Attr<
  CPred< "::llvm::isa< ::vast::core::MemberVarSymbolRefAttr >($_self)" >,
  "member variable symbol reference attribute"
> {
  let storageType = [{ ::vast::core::MemberVarSymbolRefAttr }];
  let returnType = [{ ::llvm::StringRef }];
  let valueType = NoneType;
  let constBuilderCall =
    "::mlir::SymbolRefAttr::get($_builder.getContext(), $0)";
  let convertFromStorage = "$_self.getValue()";
}

def Core_ElaboratedTypeSymbolRefAttr : Attr<
  CPred< "::llvm::isa< ::vast::core::ElaboratedTypeSymbolRefAttr >($_self)" >,
  "elaborated type symbol reference attribute"
> {
  let storageType = [{ ::vast::core::ElaboratedTypeSymbolRefAttr }];
  let returnType = [{ ::llvm::StringRef }];
  let valueType = NoneType;
  let constBuilderCall =
    "::mlir::SymbolRefAttr::get($_builder.getContext(), $0)";
  let convertFromStorage = "$_self.getValue()";
}
#endif // VAST_DIALECT_CORE_COMMON_ATTR_CONSTRAINTS_TD
