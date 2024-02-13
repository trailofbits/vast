// Copyright (c) 2024-present, Trail of Bits, Inc.

#ifndef VAST_IR_AGGREGATETYPEDEFINITIONINTERFACE
#define VAST_IR_AGGREGATETYPEDEFINITIONINTERFACE

include "mlir/IR/OpBase.td"

def AggregateTypeDefinition
    : OpInterface< "AggregateTypeDefinitionInterface" > {
        let description = [{
            This operation is encoding of a type.
        }];

        let cppNamespace = "::vast";

        let methods = [
            InterfaceMethod< "Returns element in order of their declaration.",
                "gap::generator< mlir::Type >", "getFieldTypes", (ins), [{}] >,

            InterfaceMethod< "Return all elements in order of their declaration.",
                "gap::generator< std::tuple< std::string, mlir::Type > >",
                "getFieldsInfo", (ins), [{}] >,

            InterfaceMethod< "Return all nested definitions",
                "gap::generator< vast::AggregateTypeDefinitionInterface >",
                "getNestedDeclarations", (ins), [{}] >,

            InterfaceMethod< "Get name of the defined type",
                "std::string", "getDefinedName", (ins), [{}] >
        ];
    }

#endif // VAST_IR_AGGREGATETYPEDEFINITIONINTERFACE
