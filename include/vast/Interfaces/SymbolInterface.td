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
            "llvm::StringRef", "getSymbolName", (ins), [{}],
            /*defaultImplementation=*/ [{
                auto op = this->getOperation();
                if (op->hasAttr(mlir::SymbolTable::getSymbolAttrName())) {
                    return op->template getAttrOfType<mlir::StringAttr>(
                        mlir::SymbolTable::getSymbolAttrName()
                    ).getValue();
                }
                return op->template getAttrOfType<mlir::StringAttr>("name").getValue();
            }]
        >,
        InterfaceMethod<"Sets the name of this symbol.",
            "void", "setSymbolName", (ins "llvm::StringRef":$name), [{}],
            /*defaultImplementation=*/[{
                auto op = this->getOperation();
                auto attr = mlir::StringAttr::get(op->getContext(), name);
                if (op->hasAttr(mlir::SymbolTable::getSymbolAttrName())) {
                    op->setAttr(mlir::SymbolTable::getSymbolAttrName(), attr);
                } else {
                    op->setAttr("name", attr);
                }
            }]
        >
    ];
}

#endif // VAST_DIALECT_HIGHLEVEL_IR_SYMBOLINTERFACE
