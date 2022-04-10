// Copyright (c) 2022-present, Trail of Bits, Inc.

#ifndef VAST_DIALECT_HIGHLEVEL_IR_SYMBOLINTERFACE
#define VAST_DIALECT_HIGHLEVEL_IR_SYMBOLINTERFACE

include "mlir/IR/OpBase.td"

// TODO: Add to operations
def VastSymbol : OpInterface< "VastSymbolOpInterface" > {
    let description = [{
        This interface describes an operation that may define a `VastSymbol`.
    }];
    let cppNamespace = "::vast";

    let methods = [
        InterfaceMethod<"Returns the name of this symbol.",
            "llvm::StringRef", "getName", (ins), [{}], 
            /*defaultImplementation=*/ [{
                auto name = this->getOperation()->template 
                    getAttrOfType<mlir::StringAttr>("name");
                return name.getValue();
            }]
        >,
        InterfaceMethod<"Sets the name of this symbol.",
            "void", "setName", (ins "llvm::StringRef":$name), [{}],
            /*defaultImplementation=*/[{
                this->getOperation()->setAttr(
                    "name",
                    mlir::StringAttr::get(this->getOperation()->getContext(), name)
                );
            }]
        >
    ];
}

#endif // VAST_DIALECT_HIGHLEVEL_IR_SYMBOLINTERFACE