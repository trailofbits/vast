// Copyright (c) 2021-present, Trail of Bits, Inc.

#ifndef VAST_DIALECT_HIGHLEVEL_IR_HIGHLEVELOPS
#define VAST_DIALECT_HIGHLEVEL_IR_HIGHLEVELOPS

def HighLevel_ConstantOp : HighLevel_Op< "constant", [] > {
  let summary = "VAST constant";
  let description = [{ VAST constant }];

  let arguments = (ins AnyAttr:$value);

  // FIXME: More specific type.
  let results = (outs AnyType);

  let builders = [OpBuilderDAG< (
      ins "mlir::Attribute":$value
    ) >
  ];

  let verifier = ? ;
  let printer = [{ p << getOperationName()
                     << "( " <<  value() << " )";
                   p << ": " << getResult().getType();
  }];
}

def HighLevel_ReturnOp : HighLevel_Op<"return", [Terminator]> {
  let summary = "VAST return operation.";
  let description = [{ VAST return opearation. }];

  let arguments = (ins Variadic<AnyType>:$operands);

  // Fully specified by traits.
  let verifier = ?;
}

def HighLevel_UnreachableOp : HighLevel_Op<"unreachable", [Terminator]> {
  let summary = "VAST unreachable operation";
  let description = [{ VAST unreachable operation }];

  let arguments = (ins Variadic<AnyType>:$operands);

  let builders = [OpBuilderDAG<(ins), [{ }]>];

  let verifier = ?;
  let printer = [{
    p << getOperationName() << " ";
    if (!operands().empty())
      p << *operand_begin();
  }];
}

#endif // VAST_DIALECT_HIGHLEVEL_IR_HIGHLEVELOPS
