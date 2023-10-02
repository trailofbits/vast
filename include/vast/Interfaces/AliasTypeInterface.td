// Copyright (c) 2023-present, Trail of Bits, Inc.

#ifndef VAST_IR_ALIASTYPEINTERFACE
#define VAST_IR_ALIASTYPEINTERFACE

include "mlir/IR/OpBase.td"

def AliasTypeInterface
    : TypeInterface< "AliasTypeInterface" > {
        let description = [{
            Interface to acces type alias in uniform way.
        }];

        let cppNamespace = "::vast";

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

#endif // VAST_IR_ALIASTYPEINTERFACE
