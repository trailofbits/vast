// Copyright (c) 2022-present, Trail of Bits, Inc.

#ifndef VAST_INTERFACES_SYMBOL_INTERFACE
#define VAST_INTERFACES_SYMBOL_INTERFACE

include "mlir/IR/OpBase.td"

include "vast/Dialect/Core/CoreTraits.td"
include "vast/Dialect/Core/Interfaces/Common.td"

def Core_Symbol : Core_OpInterface< "SymbolOpInterface" > {
    let description = [{
        This interface describes an operation that may define a `Symbol`.
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
        >,
        InterfaceMethod< [{
                Get all of the uses of the current symbol that are nested within
                the given operation 'from'.
                Note: See core::symbol_table::get_symbol_uses for more details.
            }],
            "::std::optional< ::mlir::SymbolTable::UseRange >", "getSymbolUses",
            (ins "::mlir::Operation *":$from), [{}],
            /*defaultImplementation=*/[{
                return ::vast::core::get_symbol_uses(this->getOperation(), from);
            }]
        >
    ];
}

def Core_VarSymbol            : Core_OpInterface< "VarSymbolOpInterface",  [Core_Symbol] >;
def Core_MemberVarSymbol      : Core_OpInterface< "MemberVarSymbolOpInterface", [Core_Symbol] >;
def Core_FuncSymbol           : Core_OpInterface< "FuncSymbolOpInterface", [Core_Symbol] >;
def Core_TypeSymbol           : Core_OpInterface< "TypeSymbolOpInterface", [Core_Symbol] >;
def Core_ElaboratedTypeSymbol : Core_OpInterface< "ElaboratedTypeSymbolOpInterface", [Core_Symbol] >;
def Core_LabelSymbol          : Core_OpInterface< "LabelSymbolOpInterface", [Core_Symbol] >;
def Core_EnumConstantSymbol   : Core_OpInterface< "EnumConstantSymbolOpInterface", [Core_Symbol] >;

#endif // VAST_INTERFACES_SYMBOL_INTERFACE
