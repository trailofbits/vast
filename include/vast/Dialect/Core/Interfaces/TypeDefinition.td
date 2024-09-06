// Copyright (c) 2022-present, Trail of Bits, Inc.

#ifndef VAST_CORE_TYPE_DEFINITION
#define VAST_CORE_TYPE_DEFINITION

include "mlir/IR/OpBase.td"

include "vast/Dialect/Core/CoreTraits.td"
include "vast/Dialect/Core/Interfaces/Common.td"

def Core_TypeDefinition : Core_OpInterface< "TypeDefinition" > {
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
            }]
        >,
        InterfaceMethod<"Returns the defined type by this operation.",
            "mlir::Type", "getDefinedType", (ins), [{}],
            /*defaultImplementation=*/ [{
                auto op = this->getOperation();
                return op->template getAttrOfType< mlir::Type >(
                    "type"
                );
            }]
        >
    ];
}

#endif // VAST_CORE_TYPE_DEFINITION
