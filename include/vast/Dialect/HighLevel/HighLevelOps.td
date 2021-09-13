// Copyright (c) 2021-present, Trail of Bits, Inc.

#ifndef VAST_DIALECT_HIGHLEVEL_IR_HIGHLEVELOPS
#define VAST_DIALECT_HIGHLEVEL_IR_HIGHLEVELOPS

def HighLevel_VarOp : HighLevel_Op< "var", [] > {
  let summary = "VAST variable declaration";
  let description = [{ VAST variable delaration }];

  let arguments = (ins
    StrAttr:$name,
    Optional<AnyType>:$initializer
  );

  let results = (outs AnyType:$result);

  let builders = [
    OpBuilderDAG< (ins "mlir::Type":$result, "llvm::StringRef":$name),
    [{
      build($_builder, $_state, result, $_builder.getStringAttr(name));
    }] >,
    OpBuilderDAG< (ins "mlir::Type":$result, "mlir::StringAttr":$name),
    [{
      build($_builder, $_state, result, name, mlir::Value());
    }] >
  ];

  let verifier = ? ;
  let printer = [{
    p << getOperationName()
      << "( " <<  name();
    if (auto init = initializer())
      p << ", " << init;
    p << " )";
    p << ": " << getResult().getType();
  }];
}


def HighLevel_DeclRefOp : HighLevel_Op< "declref", [] > {
  let summary = "VAST reference declaration";
  let description = [{ VAST reference delaration }];

  let arguments = (ins SymbolRefAttr:$decl);

  let results = (outs AnyType);

  let builders = [
    OpBuilderDAG< (ins "mlir::Type":$rty, "llvm::StringRef":$name),
    [{
      build($_builder, $_state, rty, $_builder.getSymbolRefAttr(name));
    }] >
  ];

  let printer = [{
    p << getOperationName()
      << "( " <<  decl() << " )"
      << ": " << getResult().getType();
  }];

}


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


def HighLevel_ImplicitCastOp : HighLevel_Op< "implicit_cast", [] > {
  let summary = "VAST implicit cast";
  let description = [{ VAST implicit cast }];

  let arguments = (ins AnyType:$value);
  let results = (outs AnyType);

  let verifier = ?;
  let printer = [{
    p << getOperationName()
      << " ( " << value() << " ): "
      << getResult().getType();
  }];
}


def HighLevel_AddIOp  : HighLevel_BinOp<"addi",  [Commutative]>;
def HighLevel_AddFOp  : HighLevel_BinOp<"addf",  []>;
def HighLevel_SubIOp  : HighLevel_BinOp<"subi",  []>;
def HighLevel_SubFOp  : HighLevel_BinOp<"subf",  []>;
def HighLevel_MulIOp  : HighLevel_BinOp<"muli",  [Commutative]>;
def HighLevel_MulFOp  : HighLevel_BinOp<"mulf",  []>;
def HighLevel_DivIOp  : HighLevel_BinOp<"divi",  []>;
def HighLevel_DivFOp  : HighLevel_BinOp<"divf",  []>;
def HighLevel_ModIOp  : HighLevel_BinOp<"modi",  []>;
def HighLevel_ModFOp  : HighLevel_BinOp<"modf",  []>;
def HighLevel_ModUIOp : HighLevel_BinOp<"modui", []>;


#endif // VAST_DIALECT_HIGHLEVEL_IR_HIGHLEVELOPS
