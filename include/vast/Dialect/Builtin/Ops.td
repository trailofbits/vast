// Copyright (c) 2024-present, Trail of Bits, Inc.

#ifndef VAST_DIALECT_BUILTIN_OPS
#define VAST_DIALECT_BUILTIN_OPS

include "mlir/IR/OpBase.td"

include "vast/Dialect/Core/Utils.td"

def HLBuiltin_VAStartOp : HLBuiltin_Op< "va_start" >
  , Arguments< (ins AnyType: $arg_list, AnyType: $count) >
  , Results< (outs VoidLikeType: $result) >
{
  let summary = "Start of the variable argument list";

  let assemblyFormat = "$arg_list `size` $count `:` functional-type(operands, $result) attr-dict";
}

def HLBuiltin_VAEndOp : HLBuiltin_Op< "va_end" >
  , Arguments< (ins AnyType: $arg_list) >
  , Results< (outs VoidLikeType: $result) >
{
  let summary = "End of the variable argument list";

  let assemblyFormat = "$arg_list `:` functional-type(operands, $result) attr-dict";
}

def HLBuiltin_VACopyOp : HLBuiltin_Op< "va_copy" >
  , Arguments< (ins AnyType: $src_list, AnyType: $trg_list) >
  , Results< (outs VoidLikeType: $result) >
{
  let summary = "Start of the variable argument list";

  let assemblyFormat = "$src_list `to` $trg_list `:` functional-type(operands, $result) attr-dict";
}

def HLBuiltin_TrapOp : HLBuiltin_Op< "trap" >
  , Results< (outs VoidLikeType: $result) >
{
  let summary = "Builtin trap op";
  let assemblyFormat = " `:` type($result) attr-dict";
}

def HLBuiltin_DebugTrapOp : HLBuiltin_Op< "dbgtrap" >
  , Results< (outs VoidLikeType: $result) >
{
  let summary = "Builtin debug trap op";
  let assemblyFormat = " `:` type($result) attr-dict";
}

#endif //VAST_DIALECT_BUILTIN_OPS
