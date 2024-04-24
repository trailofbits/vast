// Copyright (c) 2024, Trail of Bits, Inc.

#ifndef VAST_INTERFACES_COMMON
#define VAST_INTERFACES_COMMON

include "mlir/IR/OpBase.td"

class VastOpInterface<string name, list<Interface> baseInterfaces = []>
    : OpInterface<name, baseInterfaces>
{
    let cppNamespace = "::vast";
}

class VastTypeInterface<string name, list<Interface> baseInterfaces = []>
    : TypeInterface<name, baseInterfaces>
{
    let cppNamespace = "::vast";
}

class VastAttrInterface<string name, list<Interface> baseInterfaces = []>
    : AttrInterface<name, baseInterfaces>
{
    let cppNamespace = "::vast";
}

#endif // VAST_INTERFACES_COMMON
