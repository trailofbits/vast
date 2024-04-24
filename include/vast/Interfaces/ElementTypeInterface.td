// Copyright (c) 2023-present, Trail of Bits, Inc.

#ifndef VAST_INTERFACES_ELEMENT_TYPE_INTERFACE
#define VAST_INTERFACES_ELEMENT_TYPE_INTERFACE

include "mlir/IR/OpBase.td"
include "vast/Interfaces/Common.td"

def ElementTypeInterface : VastTypeInterface< "ElementTypeInterface" > {
    let description = [{
        Simple interface to access the element type in a uniform way.
    }];

    let methods = [
        InterfaceMethod< "Returns the element type.",
            "::mlir::Type", "getElementType", (ins),
            [{return $_type.getElementType(); }] >
    ];
}

#endif // VAST_INTERFACES_ELEMENT_TYPE_INTERFACE
