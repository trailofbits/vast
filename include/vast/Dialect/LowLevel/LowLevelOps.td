// Copyright (c) 2021-present, Trail of Bits, Inc.

#ifndef VAST_DIALECT_IR_LOWLEVELOPS
#define VAST_DIALECT_IR_LOWLEVELOPS

include "mlir/Interfaces/CallInterfaces.td"
include "mlir/IR/BuiltinAttributes.td"
include "vast/Interfaces/SymbolInterface.td"

def LowLevel_StructGEPOp
  : LowLevel_Op< "gep" >
  , Arguments<(ins AnyType:$record, I32Attr:$idx, StrAttr:$name)>
  , Results<(outs AnyType:$element)>
{
    let summary = "VAST struct gep  operation";
    let description = [{ VAST struct gep operation }];
}
#endif // VAST_DIALECT_IR_LOWLEVELOPS
