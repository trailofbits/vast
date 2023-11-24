// Copyright (c) 2023-present, Trail of Bits, Inc.

#ifndef VAST_DIALECT_HIGHLEVEL_IR_HIGHLEVELASSEMBLY
#define VAST_DIALECT_HIGHLEVEL_IR_HIGHLEVELASSEMBLY

include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinAttributes.td"

include "mlir/Interfaces/CastInterfaces.td"
include "vast/Interfaces/SymbolInterface.td"

def InlineAsmOp
  : HighLevel_Op< "inline_asm", [AttrSizedOperandSegments] >
  , Arguments<(ins
      Builtin_StringAttr:$asm_template,
      UnitAttr:$is_volatile,
      UnitAttr:$has_goto,
      //UnitAttr:$has_inline, //Clang doesn't have this qualifier?
      Variadic< AnyType >:$asm_outputs,
      Variadic< AnyType >:$asm_inputs,
      OptionalAttr< ArrayAttr >:$output_names,
      OptionalAttr< ArrayAttr >:$input_names,
      OptionalAttr< ArrayAttr >:$output_constraints,
      OptionalAttr< ArrayAttr >:$input_constraints,
      OptionalAttr< ArrayAttr >:$clobbers,
      Variadic< AnyType >:$labels
    )>
{
  let summary = "VAST operation for inline assembly";
  let description = [{ TODO }];

  let assemblyFormat = [{attr-dict $asm_template (`:`$output_names $asm_outputs^)? (`:` $output_constraints^)?  (`:`$input_names $asm_inputs^)? (`:` $input_constraints^)? (`:` $clobbers^)? (`:` $labels^)? `:` type(operands)}];
}

#endif // VAST_DIALECT_HIGHLEVEL_IR_HIGHLEVELASSEMBLY
