// Copyright (c) 2023-present, Trail of Bits, Inc.

#ifndef VAST_IR_ELEMENTTYPEINTERFACE
#define VAST_IR_ELEMENTTYPEINTERFACE

include "mlir/IR/OpBase.td"

def ElementTypeInterface
    : TypeInterface< "ElementTypeInterface" > {
        let description = [{
            Simple interface to access the element type in a uniform way.
        }];

        let cppNamespace = "::vast";

        let methods = [
            InterfaceMethod< "Returns the element type.",
                "::mlir::Type", "getElementType", (ins),
                [{return $_type.getElementType(); }] >
        ];
    }

#endif // VAST_IR_ELEMENTTYPEINTERFACE
