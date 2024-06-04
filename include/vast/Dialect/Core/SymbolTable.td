// Copyright (c) 2024, Trail of Bits, Inc.

#ifndef VAST_DIALECT_CORE_SYMBOL_TABLE
#define VAST_DIALECT_CORE_SYMBOL_TABLE

include "vast/Dialect/Core/CoreTraits.td"

class GetCppTraitName< list<Interface> interfaces >
{
    defvar cppInterfaceNames = !foreach(interface, interfaces, interface.cppInterfaceName);
    string result = !interleave(cppInterfaceNames, ", ");
}

class RecognizedSymbolsLists< list<list<Interface>> recognized_symbols_lists > {
    defvar recognized_symbols = !foreach(
        recognized_symbols_list,
        recognized_symbols_lists,
        GetCppTraitName< recognized_symbols_list >.result
    );

    string result = !interleave(recognized_symbols, ", ");
}

class RecognizedSymbolsListUsing< list<list<Interface>> recognized_symbols_lists > {
    defvar symbols_lists = RecognizedSymbolsLists< recognized_symbols_lists >.result;
    string result = "using recognized_symbols = util::type_list<" # symbols_lists # ">;";
}

class ShadowingSymbolTable< list<list<Interface>> recognized_symbols_lists >
    : CoreNativeOpTrait< "ShadowingSymbolTable", [],
        RecognizedSymbolsListUsing< recognized_symbols_lists >.result
    >
{}

#endif // VAST_DIALECT_CORE_SYMBOL_TABLE
