// Copyright (c) 2024-present, Trail of Bits, Inc.

#ifndef VAST_DIALECT_BUILTIN_OPS
#define VAST_DIALECT_BUILTIN_OPS

include "mlir/IR/OpBase.td"

def VAStartOp
  : Builtin_Op< "va_start" >
  , Arguments< (ins AnyType: $arg_list, AnyType: $count) >
  , Results< (outs AnyType: $result) >
{
  let summary = "Start of the variable argument list";

  let assemblyFormat = "$arg_list `size` $count `:` functional-type(operands, $result) attr-dict";
}

def VAEndOp : Builtin_Op< "va_end" >
  , Arguments< (ins AnyType: $arg_list) >
  , Results< (outs AnyType: $result) >
{
  let summary = "End of the variable argument list";

  let assemblyFormat = "$arg_list `:` functional-type(operands, $result) attr-dict";
}

def VACopyOp : Builtin_Op< "va_copy" >
  , Arguments< (ins AnyType: $src_list, AnyType: $trg_list) >
  , Results< (outs AnyType: $result) >
{
  let summary = "Start of the variable argument list";

  let assemblyFormat = "$src_list `to` $trg_list `:` functional-type(operands, $result) attr-dict";
}

#endif //VAST_DIALECT_BUILTIN_OPS
