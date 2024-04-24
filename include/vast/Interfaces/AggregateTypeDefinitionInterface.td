// Copyright (c) 2024-present, Trail of Bits, Inc.

#ifndef VAST_INTERFACES_AGGREGATE_TYPE_DEFINITION_INTERFACE
#define VAST_INTERFACES_AGGREGATE_TYPE_DEFINITION_INTERFACE

include "mlir/IR/OpBase.td"
include "vast/Interfaces/Common.td"

def AggregateTypeDefinition : VastOpInterface< "AggregateTypeDefinitionInterface" > {
    let description = [{
        This operation is encoding of a type.
    }];

    let methods = [
        InterfaceMethod< "Returns element in order of their declaration.",
            "gap::generator< mlir::Type >", "getFieldTypes", (ins), [{}] >,

        InterfaceMethod< "Return all elements in order of their declaration.",
            "gap::generator< vast::field_info_t >", "getFieldsInfo", (ins), [{}] >,

        InterfaceMethod< "Return all nested definitions",
            "gap::generator< vast::AggregateTypeDefinitionInterface >",
            "getNestedDeclarations", (ins), [{}] >,

        InterfaceMethod< "Get name of the defined type",
            "llvm::StringRef", "getDefinedName", (ins), [{}] >,

        InterfaceMethod< "Get defined type",
            "mlir::Type", "getDefinedType", (ins), [{}] >
    ];
}

#endif // VAST_INTERFACES_AGGREGATE_TYPE_DEFINITION_INTERFACE
