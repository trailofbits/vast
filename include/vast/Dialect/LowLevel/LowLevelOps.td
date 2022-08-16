// Copyright (c) 2021-present, Trail of Bits, Inc.

#ifndef VAST_DIALECT_IR_LOWLEVELOPS
#define VAST_DIALECT_IR_LOWLEVELOPS

include "mlir/Interfaces/CallInterfaces.td"
include "vast/Interfaces/SymbolInterface.td"

def LowLevel_GEPOp
  : LowLevel_Op< "gep" >
  , Arguments<(ins AnyType:$record, APIntAttr:$idx, StrAttr:$name)>
  , Results<(outs AnyType:$element)>
{
  let summary = "VAST gep operation";
  let description = [{ VAST gep operation }];
}
#endif // VAST_DIALECT_IR_LOWLEVELOPS
