// Copyright (c) 2023-present, Trail of Bits, Inc.

#ifndef VAST_DIALECT_HIGHLEVEL_IR_HIGHLEVELASSEMBLY
#define VAST_DIALECT_HIGHLEVEL_IR_HIGHLEVELASSEMBLY

include "mlir/IR/OpBase.td"

include "mlir/Interfaces/CastInterfaces.td"
include "vast/Interfaces/SymbolInterface.td"

def InlineAsmOp
  : HighLevel_Op< "inline_asm", [AttrSizedOperandSegments] >
  , Arguments<(ins
      AnyType:            $ass_template,
      AnyType:            $clobbers,
      Variadic< AnyType >:$outputs,
      Variadic< AnyType >:$inputs,
      Variadic< AnyType >:$labels
    )>
{
  let summary = "VAST operation for inline assembly";
  let description = [{ TODO }];
}

#endif // VAST_DIALECT_HIGHLEVEL_IR_HIGHLEVELOPS
