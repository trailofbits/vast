// Copyright (c) 2024-present, Trail of Bits, Inc.

#ifndef VAST_DIALECT_BUILTIN_OPS
#define VAST_DIALECT_BUILTIN_OPS

include "mlir/IR/OpBase.td"

def VAStartOp
  : Builtin_Op< "va.start" >
  , Arguments< (ins AnyType: $arg_list) >
{
  let summary = "Start of the variable argument list";

  let assemblyFormat = "$arg_list `:` type(operands) attr-dict";
}

def VAEndOp : Builtin_Op< "va.end" >
  , Arguments< (ins AnyType: $arg_list) >
{
  let summary = "End of the variable argument list";

  let assemblyFormat = "$arg_list `:` type(operands) attr-dict";
}

def VACopyOp : Builtin_Op< "va.copy" >
  , Arguments< (ins AnyType: $src_list, AnyType: $trg_list) >
{
  let summary = "Start of the variable argument list";

  let assemblyFormat = "$src_list `to` $trg_list `:` type(operands) attr-dict";
}

def VAArgOp : Builtin_Op< "va.arg" >
  , Arguments< (ins AnyType: $arg_list) >
  , Results< (outs AnyType: $result) >
{
  let summary = "Fetch next variadic element.";

  let assemblyFormat = "$arg_list `:` functional-type(operands, $result) attr-dict";
}

#endif //VAST_DIALECT_BUILTIN_OPS
