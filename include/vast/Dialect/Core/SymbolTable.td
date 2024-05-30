// Copyright (c) 2024, Trail of Bits, Inc.

#ifndef VAST_DIALECT_CORE_SYMBOL_TABLE
#define VAST_DIALECT_CORE_SYMBOL_TABLE

include "vast/Dialect/Core/CoreTraits.td"

class GetCppInterfaceName< list<Interface> interfaces >
{
    defvar cppInterfaceNames = !foreach(interface, interfaces, interface.cppInterfaceName);
    string result = !interleave(cppInterfaceNames, ", ");
}

class ShadowingSymbolTable< list<Interface> symbol_interfaces >
    : CoreParamNativeOpTrait< "ShadowingSymbolTable", GetCppInterfaceName< symbol_interfaces >.result >
{}

#endif // VAST_DIALECT_CORE_SYMBOL_TABLE
