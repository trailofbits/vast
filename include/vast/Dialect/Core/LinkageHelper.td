// Copyright (c) 2023, Trail of Bits, Inc.

#ifndef VAST_DIALECT_CORE_LINKAGE_HELPER
#define VAST_DIALECT_CORE_LINKAGE_HELPER
include "mlir/IR/OpBase.td"

def Core_GlobalLinkageKind : Attr<
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

#endif // VAST_DIALECT_CORE_LINKAGE_HELPER
