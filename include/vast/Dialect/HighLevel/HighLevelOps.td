// Copyright (c) 2021-present, Trail of Bits, Inc.

#ifndef VAST_DIALECT_HIGHLEVEL_IR_HIGHLEVELOPS
#define VAST_DIALECT_HIGHLEVEL_IR_HIGHLEVELOPS

// TODO(Heno): remove when NoTerminator is in mlir.
def HighLevel_ScopeEndOp : HighLevel_Op<"scope.end", [
  Terminator
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

def HighLevel_ReturnOp
  : HighLevel_Op< "return", [Terminator, NoSideEffect] >
{
  let arguments = (ins Variadic<AnyType>:$result);

  let assemblyFormat = "($result^ `:` type($result))? attr-dict ";

  // Allow building a ReturnOp with no return operand.
  let builders = [
    OpBuilderDAG<(ins), [{ build($_builder, $_state, llvm::None); }]>
  ];
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


def HighLevel_AddIOp : HighLevel_BinOp< "add", [Commutative]>;
def HighLevel_AddFOp : HighLevel_BinOp<"fadd", []>;
def HighLevel_SubIOp : HighLevel_BinOp< "sub", []>;
def HighLevel_SubFOp : HighLevel_BinOp<"fsub", []>;
def HighLevel_MulIOp : HighLevel_BinOp< "mul", [Commutative]>;
def HighLevel_MulFOp : HighLevel_BinOp<"fmul", []>;
def HighLevel_DivSOp : HighLevel_BinOp<"sdiv", []>;
def HighLevel_DivUOp : HighLevel_BinOp<"udiv", []>;
def HighLevel_DivFOp : HighLevel_BinOp<"fdiv", []>;
def HighLevel_RemSOp : HighLevel_BinOp<"srem", []>;
def HighLevel_RemUOp : HighLevel_BinOp<"urem", []>;
def HighLevel_RemFOp : HighLevel_BinOp<"frem", []>;


def HighLevel_AssignOp     : HighLevel_CompoundAssignOp< "assign", []>;
def HighLevel_AddIAssignOp : HighLevel_CompoundAssignOp< "assign.add",  []>;
def HighLevel_AddFAssignOp : HighLevel_CompoundAssignOp< "assign.fadd", []>;
def HighLevel_SubIAssignOp : HighLevel_CompoundAssignOp< "assign.sub",  []>;
def HighLevel_SubFAssignOp : HighLevel_CompoundAssignOp< "assign.fsub", []>;
def HighLevel_MulIAssignOp : HighLevel_CompoundAssignOp< "assign.mul",  []>;
def HighLevel_MulFAssignOp : HighLevel_CompoundAssignOp< "assign.fmul", []>;
def HighLevel_DivSAssignOp : HighLevel_CompoundAssignOp< "assign.sdiv", []>;
def HighLevel_DivUAssignOp : HighLevel_CompoundAssignOp< "assign.udiv", []>;
def HighLevel_DivFAssignOp : HighLevel_CompoundAssignOp< "assign.fdiv", []>;
def HighLevel_RemSAssignOp : HighLevel_CompoundAssignOp< "assign.srem", []>;
def HighLevel_RemUAssignOp : HighLevel_CompoundAssignOp< "assign.urem", []>;
def HighLevel_RemFAssignOp : HighLevel_CompoundAssignOp< "assign.frem", []>;


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


def HighLevel_IfOp : HighLevel_Op< "if", [RecursiveSideEffects] >
{
  let summary = "VAST if statement";
  let description = [{ VAST if statement }];

  let arguments = (ins Bool:$cond);
  let regions = (region SizedRegion<1>:$thenRegion, AnyRegion:$elseRegion);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilderDAG<(
      ins "Value":$cond, "bool":$withElseRegion
    )>,
    OpBuilderDAG<(
      ins "TypeRange":$result, "Value":$cond, "bool":$withElseRegion
    )>,
    OpBuilderDAG<(
      ins "TypeRange":$result, "Value":$cond,
          CArg< "BuilderCallback", "terminate_body" >:$thenBuilder,
          CArg< "BuilderCallback", "nullptr">:$elseBuilder
    )>,
    OpBuilderDAG<(
      ins "Value":$cond,
          CArg< "BuilderCallback", "terminate_body" >:$thenBuilder,
          CArg< "BuilderCallback", "nullptr">:$elseBuilder
    )>
  ];

  let printer = [{
    p << getOperationName() << " " << cond();
    p.printRegion(thenRegion());
    if (!elseRegion().empty()) {
      p << " else";
      p.printRegion(elseRegion());
    }
  }];
}


def HighLevel_WhileOp : HighLevel_Op< "while", [RecursiveSideEffects] >
{
  let summary = "VAST while statement";
  let description = [{ VAST while statement }];

  let arguments = (ins Bool:$cond);
  let regions = (region SizedRegion<1>:$body);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilderDAG<(
      ins "TypeRange":$result, "Value":$cond, CArg< "BuilderCallback", "terminate_body" >:$bodyBuilder
    )>,
    OpBuilderDAG<(
      ins "Value":$cond, CArg< "BuilderCallback", "terminate_body" >:$bodyBuilder
    )>
  ];

  let printer = [{
    p << getOperationName() << " " << cond();
    p.printRegion(body());
  }];

}

#endif // VAST_DIALECT_HIGHLEVEL_IR_HIGHLEVELOPS
