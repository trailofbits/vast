// Copyright (c) 2024, Trail of Bits, Inc.

#ifndef VAST_INTERFACES_ANALYSES_COMMON
#define VAST_INTERFACES_ANALYSES_COMMON

include "mlir/IR/OpBase.td"

class VAST_AnalysesOpInterface<string name, list<Interface> baseInterfaces = []>
    : OpInterface<name, baseInterfaces>
{
    let cppNamespace = "::vast::analyses";
}

class VAST_AnalysesTypeInterface<string name, list<Interface> baseInterfaces = []>
    : TypeInterface<name, baseInterfaces>
{
    let cppNamespace = "::vast::analyses";
}

class VAST_AnalysesAttrInterface<string name, list<Interface> baseInterfaces = []>
    : AttrInterface<name, baseInterfaces>
{
    let cppNamespace = "::vast::analyses";
}

#endif // VAST_INTERFACES_ANALYSES_COMMON

