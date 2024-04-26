// Copyright (c) 2023-present, Trail of Bits, Inc.

#ifndef VAST_INTERFACES_DEFAULT_DATA_LAYOUT_TYPE_INTERFACE
#define VAST_INTERFACES_DEFAULT_DATA_LAYOUT_TYPE_INTERFACE

include "mlir/IR/OpBase.td"
include "vast/Interfaces/Common.td"

def DefaultDataLayoutTypeInterface : VastTypeInterface< "DefaultDataLayoutTypeInterface" > {
    let description = [{
        Default data layout type interface implimentation.
    }];

    let methods = [
        InterfaceMethod<
            /*description=*/"Returns the size of the given type in bits computed "
                            "using the relevant entries. The data layout object can "
                            "be used for recursive queries.",
            /*retTy=*/"llvm::TypeSize",
            /*methodName=*/"getTypeSizeInBits",
            /*args=*/(ins "const ::mlir::DataLayout &":$dataLayout,
                            "::mlir::DataLayoutEntryListRef":$params),
            /*methodBody=*/"",
            /*defaultImplementation=*/[{
                auto get = [](const auto &entry) { return entry.bw; };
                return default_dl_query< ConcreteType, llvm::TypeSize >(
                    *this, get, dataLayout, params);
            }]
        >,
        InterfaceMethod<
            /*description=*/"Returns the ABI-required alignment for this type, "
                            "in bytes",
            /*retTy=*/"unsigned",
            /*methodName=*/"getABIAlignment",
            /*args=*/(ins "const ::mlir::DataLayout &":$dataLayout,
                            "::mlir::DataLayoutEntryListRef":$params),
            /*methodBody=*/"",
            /*defaultImplementation=*/[{
                auto get = [](const auto &entry) { return entry.abi_align; };
                return default_dl_query< ConcreteType, unsigned >(
                    *this, get, dataLayout, params);
            }]
        >,
        InterfaceMethod<
            /*description=*/"Returns the preferred alignment for this type, "
                            "in bytes.",
            /*retTy=*/"llvm::TypeSize",
            /*methodName=*/"getPreferredAlignment",
            /*args=*/(ins "const ::mlir::DataLayout &":$dataLayout,
                            "::mlir::DataLayoutEntryListRef":$params),
            /*methodBody=*/"",
            /*defaultImplementation=*/[{
                VAST_UNIMPLEMENTED;
            }]
        >,
    ];
}

#endif // VAST_INTERFACES_DEFAULT_DATA_LAYOUT_TYPE_INTERFACE
