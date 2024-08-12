// Copyright (c) 2024, Trail of Bits, Inc.

#ifndef VAST_INTERFACES_AST_COMMON
#define VAST_INTERFACES_AST_COMMON

include "mlir/IR/OpBase.td"

class VAST_ASTOpInterface<string name, list<Interface> baseInterfaces = []>
    : OpInterface<name, baseInterfaces>
{
    let cppNamespace = "::vast::ast";
}

class VAST_ASTTypeInterface<string name, list<Interface> baseInterfaces = []>
    : TypeInterface<name, baseInterfaces>
{
    let cppNamespace = "::vast::ast";
}

class VAST_ASTAttrInterface<string name, list<Interface> baseInterfaces = []>
    : AttrInterface<name, baseInterfaces>
{
    let cppNamespace = "::vast::ast";
}

#endif // VAST_INTERFACES_AST_COMMON

