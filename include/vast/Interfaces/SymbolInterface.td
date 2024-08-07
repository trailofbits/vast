// Copyright (c) 2022-present, Trail of Bits, Inc.

#ifndef VAST_INTERFACES_SYMBOL_INTERFACE
#define VAST_INTERFACES_SYMBOL_INTERFACE

include "mlir/IR/OpBase.td"
include "vast/Interfaces/Common.td"
include "vast/Dialect/Core/CoreTraits.td"

def VastSymbol : VastOpInterface< "VastSymbolOpInterface" > {
    let description = [{
        This interface describes an operation that may define a `VastSymbol`.
    }];

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

def VarSymbol            : VastOpInterface< "VarSymbolOpInterface",  [VastSymbol] >;
def MemberVarSymbol      : VastOpInterface< "MemberVarSymbolOpInterface", [VastSymbol] >;
def FuncSymbol           : VastOpInterface< "FuncSymbolOpInterface", [VastSymbol] >;
def TypeSymbol           : VastOpInterface< "TypeSymbolOpInterface", [VastSymbol] >;
def ElaboratedTypeSymbol : VastOpInterface< "ElaboratedTypeSymbolOpInterface", [VastSymbol] >;
def LabelSymbol          : VastOpInterface< "LabelSymbolOpInterface", [VastSymbol] >;
def EnumConstantSymbol   : VastOpInterface< "EnumConstantSymbolOpInterface", [VastSymbol] >;

#endif // VAST_INTERFACES_SYMBOL_INTERFACE
