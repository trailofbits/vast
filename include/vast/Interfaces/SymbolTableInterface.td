// Copyright (c) 2022-present, Trail of Bits, Inc.

#ifndef VAST_INTERFACES_SYMBOL_TABLE_INTERFACE
#define VAST_INTERFACES_SYMBOL_TABLE_INTERFACE

include "mlir/IR/OpBase.td"
include "vast/Interfaces/Common.td"
include "vast/Dialect/Core/CoreTraits.td"

def SymbolTableOpInterface : VastOpInterface< "SymbolTableOpInterface" > {
    let methods = [
        InterfaceMethod<"Returns true if the symbol table holds symbols that have the given symbol kind trait.",
            "bool", "can_hold_symbol_kind", (ins "core::symbol_kind":$kind), [{
                return core::can_hold_symbol_kind< typename ConcreteOp::recognized_symbols_lists >(kind);
            }]
        >,
        InterfaceMethod<"Returns true if the symbol table can hold operation (i.e. opetaion declares recognized symbol kind).",
            "bool", "can_hold_operation", (ins "mlir::Operation *":$op), [{
                return core::can_hold_operation< typename ConcreteOp::recognized_symbols_lists >(op);
            }]
        >,
        InterfaceMethod<"Returns recognized symbols.",
            "llvm::SmallVector< vast::core::symbol_kind >", "symbol_kinds", (ins), [{
                return core::symbol_kinds< typename ConcreteOp::recognized_symbols_lists >();
            }]
        >,
        InterfaceMethod<"Initializes the symbol table based on the recognized symbols lists.",
            "::vast::core::symbol_table", "materialize", (ins), [{
                return ::vast::core::symbol_table(std::in_place_type<
                    typename ConcreteOp::recognized_symbols_lists >, $_op
                );
            }]
        >
    ];
}

class GetCppTraitName< list<Interface> interfaces >
{
    defvar cppInterfaceNames = !foreach(interface, interfaces, interface.cppInterfaceName);
    string result = [{ util::type_list< }] # !interleave(cppInterfaceNames, ", ") # [{ > }];
}

class RecognizedSymbolsLists< list< list<Interface> > recognized_symbols_lists > {
    defvar recognized_symbols = !foreach(
        recognized_symbols_list,
        recognized_symbols_lists,
        GetCppTraitName< recognized_symbols_list >.result
    );

    string result = !interleave(recognized_symbols, ", ");
}

class RecognizedSymbols< list< list<Interface> > recognized_symbols_lists > {
    defvar symbols_lists = RecognizedSymbolsLists< recognized_symbols_lists >.result;
    string result = [{
        using recognized_symbols_lists = util::type_list< }] # symbols_lists # [{ >;

        static_assert( util::is_unique< util::flatten< recognized_symbols_lists > >,
            "Symbol tables needs to recognize unique symbol kinds."
        );
    }];
}

class ShadowingSymbolTableTrait< list< list<Interface> > recognized_symbols_lists >
    : CoreNativeOpTrait< "ShadowingSymbolTable", [],
        RecognizedSymbols< recognized_symbols_lists >.result
    >
{}

class ShadowingSymbolTable< list< list<Interface> > recognized_symbols_lists >
    : Interface< "ShadowingSymbolTable", [SymbolTableOpInterface] >
    , ShadowingSymbolTableTrait< recognized_symbols_lists >
{}

#endif // VAST_INTERFACES_SYMBOL_TABLE_INTERFACE
