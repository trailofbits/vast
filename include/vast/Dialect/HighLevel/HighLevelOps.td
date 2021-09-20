// Copyright (c) 2021-present, Trail of Bits, Inc.

#ifndef VAST_DIALECT_HIGHLEVEL_IR_HIGHLEVELOPS
#define VAST_DIALECT_HIGHLEVEL_IR_HIGHLEVELOPS

// TODO(Heno): remove when NoTerminator is in mlir.
def HighLevel_ScopeEndOp : HighLevel_Op<"scope-end", [
  Terminator, HasParent<"ScopeOp">
]> {
  let summary = "A pseudo op that marks the end of a hl.scope";
  let description = [{
    This op terminates the only block inside the only region of a `hl.scope`.
  }];

  let parser = [{ return mlir::success(); }];
  let printer = [{ p << getOperationName(); }];
}
def HighLevel_ScopeOp
  : HighLevel_Op< "scope", [] >
{
  let arguments = (ins);
  let results = (outs);
  let regions = (region AnyRegion:$body);

  let printer = [{
    p << getOperationName();
    p.printRegion(body(), false /* print entry blocks */);
  }];
}

def HighLevel_VarOp : HighLevel_Op< "var", [] > {
  let summary = "VAST variable declaration";
  let description = [{ VAST variable declaration }];

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
  let description = [{ VAST reference declaration }];

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


def HighLevel_VoidOp : HighLevel_Op<"void", []> {
  let summary = "VAST void operation";
  let description = [{ VAST void operation }];

  let results = (outs Void);

  let builders = [
    OpBuilderDAG<(ins),
    [{
      build($_builder, $_state, VoidType::get($_builder.getContext()));
    }] >
  ];

  let verifier = ?;
  let printer = [{ p << getOperationName(); }];
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
      << "( " << value() << " ): "
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


class PredicateAttr< string name, int val > : I64EnumAttrCase< name, val > {}

class PredicateList< string name, string summary, list<PredicateAttr> cases >
  : I64EnumAttr< name, summary, cases > {}

def ICmpPredicateEQ  : PredicateAttr< "eq", 0>;
def ICmpPredicateNE  : PredicateAttr< "ne", 1>;
def ICmpPredicateSLT : PredicateAttr<"slt", 2>;
def ICmpPredicateSLE : PredicateAttr<"sle", 3>;
def ICmpPredicateSGT : PredicateAttr<"sgt", 4>;
def ICmpPredicateSGE : PredicateAttr<"sge", 5>;
def ICmpPredicateULT : PredicateAttr<"ult", 6>;
def ICmpPredicateULE : PredicateAttr<"ule", 7>;
def ICmpPredicateUGT : PredicateAttr<"ugt", 8>;
def ICmpPredicateUGE : PredicateAttr<"uge", 9>;

let cppNamespace = "::vast::hl" in
def Predicate : PredicateList< "Predicate", "comparison predicate", [
  ICmpPredicateEQ,  ICmpPredicateNE,  ICmpPredicateSLT, ICmpPredicateSLE,
  ICmpPredicateSGT, ICmpPredicateSGE, ICmpPredicateULT, ICmpPredicateULE,
  ICmpPredicateUGT, ICmpPredicateUGE
] >;


def HighLevel_CmpOp : HighLevel_Op< "cmp", [NoSideEffect, SameTypeOperands] >
{
  let summary = "VAST comparison operation";
  let description = [{ VAST comparison operation }];

  let arguments = (ins Predicate:$predicate, AnyType:$lhs, AnyType:$rhs);
  let results = (outs Bool:$result);

  let assemblyFormat = "$predicate $lhs `,` $rhs  attr-dict `:` type($lhs)";

  let builders = [
    OpBuilderDAG<(ins "Predicate":$pred, "Value":$lhs, "Value":$rhs), [{
      auto boolty = BoolType::get($_builder.getContext());
      return build($_builder, $_state, boolty, pred, lhs, rhs);
    }]>
  ];
}

#endif // VAST_DIALECT_HIGHLEVEL_IR_HIGHLEVELOPS
