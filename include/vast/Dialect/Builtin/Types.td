// Copyright (c) 2024-present, Trail of Bits, Inc.

#ifndef VAST_DIALECT_BUILTINTYPES
#define VAST_DIALECT_BUILTINTYPES

include "mlir/Interfaces/DataLayoutInterfaces.td"
include "mlir/IR/BuiltinTypeInterfaces.td"

class BuiltinType< string name, string mnem, list< Trait > traits = [] >
  : TypeDef< Builtin_Dialect, name, traits >
{
  let mnemonic = mnem;
}

def VaList : BuiltinType< "VaList", "va_list" >
{

}

#endif // VAST_DIALECT_BUILTINTYPES
