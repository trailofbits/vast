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
                /*retTy=*/"unsigned",
                /*methodName=*/"getTypeSizeInBits",
                /*args=*/(ins "const ::mlir::DataLayout &":$dataLayout,
                              "::mlir::DataLayoutEntryListRef":$params),
                /*methodBody=*/"",
                /*defaultImplementation=*/[{
                    VAST_CHECK(params.size() != 0,
                        "Query for getTypeSizeInBits failed: Must have at least one entry!",
                    );

                    std::optional<uint32_t> out;
                    auto handle_param = [&](auto &entry) {
                        if (!out) out = entry.bw;
                        VAST_CHECK(*out == entry.bw,
                            "Inconsistent entries {0}, {1}, entry count: {2}",
                            *out, entry.bw, params.size()
                        );
                    };
                    for (const auto &param : params) {
                        auto entry = dl::DLEntry::unwrap(param);
                        if (mlir::isa< ConcreteType >(entry.type)) {
                            handle_param(entry);
                        }
                    }
                    VAST_CHECK(out.has_value(), "getTypeSizeBits entries did not yield result.");
                    return *out;
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
                    VAST_UNIMPLEMENTED;
                }]
            >,
            InterfaceMethod<
                /*description=*/"Returns the preferred alignment for this type, "
                                "in bytes.",
                /*retTy=*/"unsigned",
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
