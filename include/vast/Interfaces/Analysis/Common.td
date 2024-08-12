// Copyright (c) 2024, Trail of Bits, Inc.

#ifndef VAST_INTERFACES_ANALYSIS_COMMON
#define VAST_INTERFACES_ANALYSIS_COMMON

include "mlir/IR/OpBase.td"

class VAST_AnalysisOpInterface<string name, list<Interface> baseInterfaces = []>
    : OpInterface<name, baseInterfaces>
{
    let cppNamespace = "::vast::analysis";
}

class VAST_AnalysisTypeInterface<string name, list<Interface> baseInterfaces = []>
    : TypeInterface<name, baseInterfaces>
{
    let cppNamespace = "::vast::analysis";
}

class VAST_AnalysisAttrInterface<string name, list<Interface> baseInterfaces = []>
    : AttrInterface<name, baseInterfaces>
{
    let cppNamespace = "::vast::analysis";
}

#endif // VAST_INTERFACES_ANALYSIS_COMMON
