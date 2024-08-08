// Copyright (c) 2024, Trail of Bits, Inc.

#ifndef VAST_INTERFACES_CFG_COMMON
#define VAST_INTERFACES_CFG_COMMON

include "mlir/IR/OpBase.td"

class VAST_CFGOpInterface<string name, list<Interface> baseInterfaces = []>
    : OpInterface<name, baseInterfaces>
{
    let cppNamespace = "::vast::cfg";
}

class VAST_CFGTypeInterface<string name, list<Interface> baseInterfaces = []>
    : TypeInterface<name, baseInterfaces>
{
    let cppNamespace = "::vast::cfg";
}

class VAST_CFGAttrInterface<string name, list<Interface> baseInterfaces = []>
    : AttrInterface<name, baseInterfaces>
{
    let cppNamespace = "::vast::cfg";
}

#endif // VAST_INTERFACES_CFG_COMMON
