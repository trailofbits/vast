// Copyright (c) 2022-present, Trail of Bits, Inc.

#ifndef VAST_CORE_TYPE_DEFINITION
#define VAST_CORE_TYPE_DEFINITION

include "mlir/IR/OpBase.td"

include "vast/Dialect/Core/CoreTraits.td"
include "vast/Dialect/Core/Interfaces/Common.td"

def Core_TypeDefinition : Core_OpInterface< "TypeDefinitionInterface" > {
    let description = [{
        This interface allows to retrieve defined type by an operations and its name.
    }];
    let methods = [
        InterfaceMethod<"Returns the name of defined type.",
            "llvm::StringRef", "getDefinedName", (ins), [{}],
            /*defaultImplementation=*/ [{
                auto op = this->getOperation();
                if (op->hasAttr(mlir::SymbolTable::getSymbolAttrName())) {
                    return op->template getAttrOfType< mlir::StringAttr >(
                        mlir::SymbolTable::getSymbolAttrName()
                    ).getValue();
                }

                VAST_FATAL("Operation does not have a defined name.");
            }]
        >,
        InterfaceMethod<"Returns the defined type by this operation.",
            "mlir::Type", "getDefinedType", (ins), [{}],
            /*defaultImplementation=*/ [{
                auto op = this->getOperation();
                return op->template getAttrOfType< mlir::TypeAttr >(
                    "type"
                ).getValue();
            }]
        >
    ];
}

def Core_AggregateTypeDefinition : Core_OpInterface<
    "AggregateTypeDefinitionInterface", [Core_TypeDefinition]
> {
    let description = [{
        This operation is encoding of a type.
    }];

    let methods = [
        InterfaceMethod< "Returns element in order of their declaration.",
            "gap::generator< mlir::Type >", "getFieldTypes", (ins), [{}] >,

        InterfaceMethod< "Return all elements in order of their declaration.",
            "gap::generator< vast::core::field_info_t >", "getFieldsInfo", (ins), [{}] >,

        InterfaceMethod< "Return all nested definitions",
            "gap::generator< vast::core::AggregateTypeDefinitionInterface >",
            "getNestedDeclarations", (ins), [{}] >,
    ];
}

#endif // VAST_CORE_TYPE_DEFINITION
