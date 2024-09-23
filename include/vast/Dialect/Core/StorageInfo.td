#ifndef VAST_DIALECT_CORE_STORAGEINFO
#define VAST_DIALECT_CORE_STORAGEINFO

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"

def Core_StorageClass : Attr<
    CPred<"::mlir::isa<::vast::core::StorageClassAttr>($_self)">,
    "declaration storage class"
> {
    let storageType = "::vast::core::StorageClassAttr";
    let returnType  = "::vast::core::StorageClass";
    let convertFromStorage = "$_self.getValue()";
    let constBuilderCall = [{
        ::vast::core::StorageClassAttr::get($_builder.getContext(), $0)
    }];
}

def Core_ThreadStorage : Attr<
    CPred<"::mlir::isa<::vast::core::TSClassAttr>($_self)">,
    "declaration storage class"
> {
    let storageType = "::vast::core::TSClassAttr";
    let returnType  = "::vast::core::TSClass";
    let convertFromStorage = "$_self.getValue()";
    let constBuilderCall = [{
        ::vast::core::TSClassAttr::get($_builder.getContext(), $0)
    }];
}

#endif //VAST_DIALECT_CORE_STORAGEINFO
