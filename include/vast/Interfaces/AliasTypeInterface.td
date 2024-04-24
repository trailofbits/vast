// Copyright (c) 2023-present, Trail of Bits, Inc.

#ifndef VAST_INTERFACES_ALIAS_TYPE_INTERFACE
#define VAST_INTERFACES_ALIAS_TYPE_INTERFACE

include "mlir/IR/OpBase.td"
include "vast/Interfaces/Common.td"

def AliasTypeInterface : VastTypeInterface< "AliasTypeInterface" > {
    let description = [{
        Interface to acces type alias in uniform way.
    }];

    let methods = [
        InterfaceMethod< "Returns type alias.",
            "llvm::StringRef", "getAlias", (ins), "",
            [{ return $_type.getMnemonic(); }]
        >,
        InterfaceMethod< "Returns alias result kind.",
            "mlir::OpAsmDialectInterface::AliasResult", "getAliasResultKind", (ins), "",
            [{ return mlir::OpAsmDialectInterface::AliasResult::OverridableAlias; }]
        >
    ];
}

#endif // VAST_INTERFACES_ALIAS_TYPE_INTERFACE
