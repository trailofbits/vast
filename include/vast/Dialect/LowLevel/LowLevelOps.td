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

def UninitializedVar
    : LowLevel_Op< "unintialized_var", [VastSymbol, NoSideEffect] >
    , Results<(outs AnyType:$result)>
{
    let summary = "Declaration of variable that have not been initialized yet.";
    let description = [{ Declaration of variable that have not been initialized yet. }];

    let results = (outs AnyType:$result);
}

def InitializeVar
    : LowLevel_Op< "initialize", [NoSideEffect] >
    , Arguments<(ins AnyType:$var, Variadic<AnyType>:$elements)>
    , Results<(outs AnyType:$result)>
{
    let summary = "Initialize a variable.";
    let description = [{
        Initialize a variable - for now this operation is a direct lowering from hl.var
        initialization section. Later there will be need to discover how this ties
        to constructors.
    }];
}
#endif // VAST_DIALECT_IR_LOWLEVELOPS
