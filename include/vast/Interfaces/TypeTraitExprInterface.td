// Copyright (c) 2024-present, Trail of Bits, Inc.

#ifndef VAST_INTERFACES_TYPE_TRAIT_EXPR_INTERFACE
#define VAST_INTERFACES_TYPE_TRAIT_EXPR_INTERFACE

include "mlir/IR/OpBase.td"
include "vast/Interfaces/Common.td"

def TypeTraitExprInterface : VastOpInterface< "TypeTraitExprInterface" > {
    let description = [{
        Interface to use type trait expressions uniformly
    }];

    let methods = [
        InterfaceMethod< "Returns type trait arguments.",
            "llvm::SmallVector< mlir::Type >", "getArgs", (ins), ""
        >,
        InterfaceMethod< "Returns trait value, if possible.",
            "std::optional< bool >", "getValue", (ins), ""
        >
    ];
}

#endif // VAST_INTERFACES_TYPE_TRAIT_EXPR_INTERFACE
