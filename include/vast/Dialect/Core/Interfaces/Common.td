// Copyright (c) 2024, Trail of Bits, Inc.

#ifndef VAST_DIALECT_CORE_INTERFACES_COMMON
#define VAST_DIALECT_CORE_INTERFACES_COMMON

include "mlir/IR/OpBase.td"

class Core_OpInterface<string name, list<Interface> baseInterfaces = []>
    : OpInterface<name, baseInterfaces>
{
    let cppNamespace = "::vast::core";
}

class Core_TypeInterface<string name, list<Interface> baseInterfaces = []>
    : TypeInterface<name, baseInterfaces>
{
    let cppNamespace = "::vast::core";
}

class Core_AttrInterface<string name, list<Interface> baseInterfaces = []>
    : AttrInterface<name, baseInterfaces>
{
    let cppNamespace = "::vast::core";
}

#endif // VAST_DIALECT_CORE_INTERFACES_COMMON
