// Copyright (c) 2022-present, Trail of Bits, Inc.

#ifndef VAST_INTERFACES_SYMBOL_TABLE_INTERFACE
#define VAST_INTERFACES_SYMBOL_TABLE_INTERFACE

include "mlir/IR/OpBase.td"
include "vast/Interfaces/Common.td"
include "vast/Dialect/Core/CoreTraits.td"

include "vast/Dialect/Core/Interfaces/Common.td"

def VastCallOpInterface : VastOpInterface< "VastCallOpInterface" > {
    let methods = [
        InterfaceMethod<[{
            This is temporal interface method. It will be removed
            when changes of `mlir::CallOpInterface` are upstreamed.
        }],
        "::mlir::Operation *", "resolveCallableInTable", (ins "::mlir::SymbolTableCollection &":$symbolTable)
        >,
        InterfaceMethod<[{
            This is temporal interface method. It will be removed
            when changes of `mlir::CallOpInterface` are upstreamed.
        }],
        "::mlir::Operation *", "resolveCallable"
        >
    ];
}

def Core_SymbolTableOpInterface : Core_OpInterface< "SymbolTableOpInterface" > {
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

class Core_GetCppTraitName< list<Interface> interfaces >
{
    defvar cppInterfaceNames = !foreach(interface, interfaces, interface.cppNamespace # "::" # interface.cppInterfaceName);
    string result = [{ util::type_list< }] # !interleave(cppInterfaceNames, ", ") # [{ > }];
}

class Core_RecognizedSymbolsLists< list< list<Interface> > recognized_symbols_lists > {
    defvar recognized_symbols = !foreach(
        recognized_symbols_list,
        recognized_symbols_lists,
        Core_GetCppTraitName< recognized_symbols_list >.result
    );

    string result = !interleave(recognized_symbols, ", ");
}

class Core_RecognizedSymbols< list< list<Interface> > recognized_symbols_lists > {
    defvar symbols_lists = Core_RecognizedSymbolsLists< recognized_symbols_lists >.result;
    string result = [{
        using recognized_symbols_lists = util::type_list< }] # symbols_lists # [{ >;

        static_assert( util::is_unique< util::flatten< recognized_symbols_lists > >,
            "Symbol tables needs to recognize unique symbol kinds."
        );
    }];
}

class Core_EmptySymbolTableTrait
    : Core_NativeOpTrait< "EmptySymbolTable", [], [{
        using recognized_symbols_lists = util::type_list<>;
    }] >
{}

def Core_EmptySymbolTable
    : Interface< "EmptySymbolTable", [Core_SymbolTableOpInterface] >
    , Core_EmptySymbolTableTrait
{}

class Core_ShadowingSymbolTableTrait< list< list<Interface> > recognized_symbols_lists >
    : Core_NativeOpTrait< "ShadowingSymbolTable", [],
        Core_RecognizedSymbols< recognized_symbols_lists >.result
    >
{}

class Core_ShadowingSymbolTable< list< list<Interface> > recognized_symbols_lists >
    : Interface< "ShadowingSymbolTable", [Core_SymbolTableOpInterface] >
    , Core_ShadowingSymbolTableTrait< recognized_symbols_lists >
{}

#endif // VAST_INTERFACES_SYMBOL_TABLE_INTERFACE
