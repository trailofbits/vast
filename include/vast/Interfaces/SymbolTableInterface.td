// Copyright (c) 2022-present, Trail of Bits, Inc.

#ifndef VAST_INTERFACES_SYMBOL_TABLE_INTERFACE
#define VAST_INTERFACES_SYMBOL_TABLE_INTERFACE

include "mlir/IR/OpBase.td"
include "vast/Interfaces/Common.td"
include "vast/Dialect/Core/CoreTraits.td"

def VastSymbolTable : VastOpInterface< "VastSymbolTable" > {
    let methods = [
        InterfaceMethod<"TODO",
            "bool", "holds_symbol_kind", (ins "core::symbol_kind":$kind), [{
                return core::holds_symbol_kind<
                    util::flatten< typename ConcreteOp::recognized_symbols_lists >
                >(kind);
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
    : Interface< "ShadowingSymbolTable", [VastSymbolTable] >
    , ShadowingSymbolTableTrait< recognized_symbols_lists >
{}

#endif // VAST_INTERFACES_SYMBOL_TABLE_INTERFACE
