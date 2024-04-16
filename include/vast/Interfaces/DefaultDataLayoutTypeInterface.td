// Copyright (c) 2023-present, Trail of Bits, Inc.

#ifndef VAST_IR_DEFAULTDATALAYOUTTYPEINTERFACE
#define VAST_IR_DEFAULTDATALAYOUTTYPEINTERFACE

include "mlir/IR/OpBase.td"

def DefaultDataLayoutTypeInterface
    : TypeInterface< "DefaultDataLayoutTypeInterface" > {
        let description = [{
            Default data layout type interface implimentation.
        }];

        let cppNamespace = "::vast";

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

#endif // VAST_IR_DEFAULTDATALAYOUTTYPEINTERFACE
