// Copyright (c) 2021-present, Trail of Bits, Inc.

#ifndef VAST_DIALECT_HIGHLEVEL_IR_HIGHLEVELCF
#define VAST_DIALECT_HIGHLEVEL_IR_HIGHLEVELCF

include "mlir/Interfaces/ControlFlowInterfaces.td"

include "vast/Dialect/Core/CoreTraits.td"

class ControlFlowOp< string mnemonic, list< Trait > traits = [] >
    : HighLevel_Op< mnemonic, !listconcat(traits,
        [NoTerminator, NoRegionArguments, DeclareOpInterfaceMethods<RegionKindInterface>]
      ) >
{
    let summary = "VAST control flow operation";
    let description = [{ VAST control flow operation }];
}

def HighLevel_CondYieldOp : HighLevel_Op< "cond.yield", [
  // TODO(Heno): add ReturnLike trait
  Terminator, ParentOneOf<["IfOp", "WhileOp", "ForOp", "DoOp", "CondOp"]>
] > {
  let summary = "condition yield operation";
  let description = [{
      A condition yield operation is used to terminate the region representing
      condition expression of control flow operations `IfOp`, `WhileOp`, `ForOp`
      and `DoOp`. It yields a boolean value for the conditional branch.

      The custom assembly form of the operation is as follows:

      hl.cond.yield result : BoolType
  }];

  let arguments = (ins AnyType:$result);
  let results = (outs);

  let assemblyFormat = [{ attr-dict $result `:` type($result) }];
}

def HighLevel_ValueYieldOp : HighLevel_Op< "value.yield", [
  // TODO(Heno): add ReturnLike trait
  Terminator
] > {
  let summary = "value yield operation";
  let description = [{
      A value yield operation is used to terminate the case region of a switch
      statement. The yielded value triggers the parent case statement region.

      The custom assembly form of the operation is as follows:

      hl.value.yield result : type
   }];

  // TODO(Heno): check on type
  let arguments = (ins AnyType:$result);

  let assemblyFormat = [{ attr-dict $result `:` type($result) }];
}

def HighLevel_TypeYieldOp : HighLevel_Op< "type.yield", [
  // TODO(Heno): add ReturnLike trait
  Terminator
] > {
  let summary = "type yield operation";
  let description = [{
      A type yield operation is used to terminate the underlying expression
      region of a typeof(expr) statement.

      The custom assembly form of the operation is as follows:

      hl.type.yield result : type
   }];

  // TODO(Heno): check on type
  let arguments = (ins AnyType:$result);

  let extraClassDeclaration = [{ mlir::Type getYielded() {return getResult().getType();} }];

  let assemblyFormat = [{ attr-dict $result `:` type($result) }];
}

def HighLevel_IfOp : ControlFlowOp< "if" >
{
  let summary = "VAST if statement";
  let description = [{
    The operation takes builders of two mandatory regions -- condition and then
    region -- and one builder optional region representing else block of C if statement.
    Builders, given the location, build a particular region.

    The generic form of the operation is as follows:

    hl.if {
      ... /* condition region */
      hl.cond.yield %cond : !hl.bool
    } then {
      ... /* then region */
    } else {
      ... /* else region */
    }
  }];

  let regions = (region CondRegion:$condRegion, AnyRegion:$thenRegion, AnyRegion:$elseRegion);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins
      "BuilderCallback":$condBuilder,
      "BuilderCallback":$thenBuilder,
      CArg<"BuilderCallback", "std::nullopt">:$elseBuilder
    )>
  ];

  let extraClassDeclaration = [{
    /// Returns true if an else block exists.
    bool hasElse() { return !getElseRegion().empty(); }
  }];

  let hasCustomAssemblyFormat = 1;
}

def HighLevel_CondOp : ControlFlowOp< "cond" >
    , Results<(outs AnyType:$result)>
{
  let summary = "VAST conditional statement";
  let description = [{
    The operation takes builders of three regions -- condition, true branch and false branch.
    Builders, given the location, build a particular region.

    The generic form of the operation is as follows:

    hl.cond {
      ... /* condition region */
      hl.cond.yield %cond : !hl.bool
    } ? {
      ... /* true region */
    } : {
      ... /* false region */
    }
  }];

  let regions = (region
    CondRegion:$condRegion,
    ValueRegion:$thenRegion,
    ValueRegion:$elseRegion
  );

  let hasRegionVerifier = 1;

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins
      "Type":$type,
      "BuilderCallback":$condBuilder,
      "BuilderCallback":$thenBuilder,
      "BuilderCallback":$elseBuilder
    )>
  ];

  let extraClassDeclaration = [{
    bool typesMatch(mlir::Type lhs, mlir::Type rhs);
  }];

  let assemblyFormat = [{ $condRegion `?` $thenRegion `:` $elseRegion attr-dict `:` type(results) }];
}

def HighLevel_WhileOp : ControlFlowOp< "while", [NoTerminator] >
{
  let summary = "VAST while statement";
  let description = [{
    The operation takes builders of two mandatory regions -- condition and body
    region. Builders, given the location, build a particular region.

    The generic form of the operation is as follows:

    hl.while {
      ... /* condition region */
      hl.cond.yield %cond : !hl.bool
    } do {
      ... /* body region */
    }
  }];

  let regions = (region CondRegion:$condRegion, AnyRegion:$bodyRegion);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins
      "BuilderCallback":$condBuilder,
      "BuilderCallback":$bodyBuilder
    )>
  ];

  let assemblyFormat = [{ $condRegion `do` $bodyRegion attr-dict }];
}


def HighLevel_ForOp : ControlFlowOp< "for" >
{
  let summary = "VAST for statement";
  let description = [{
    Operation represents a for-loop statement.

    The generic form of the operation is as follows:

    hl.for {
      ... /* cond region */
      hl.cond.yield %cond : !hl.bool
    } incr {
      ... /* increment/update region */
    } do {
      ... /* body region */
    }
  }];

  let regions = (region
    CondRegion:$condRegion,
    AnyRegion:$incrRegion,
    AnyRegion:$bodyRegion
  );

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins
      "BuilderCallback":$condBuilder,
      "BuilderCallback":$incrBuilder,
      "BuilderCallback":$bodyBuilder
    )>
  ];

  let assemblyFormat = [{
    $condRegion `incr` $incrRegion attr-dict `do` $bodyRegion
  }];
}

def HighLevel_DoOp : ControlFlowOp< "do" >
{
  let summary = "VAST do-while statement";
  let description = [{
    The operation represents a do-while statement.

    The generic form of the operation is as follows:

    hl.do {
      ... /* body region */
    } cond {
      ... /* cond region */
      hl.cond.yield %cond : !hl.bool
    }
  }];

  let regions = (region AnyRegion:$bodyRegion, CondRegion:$condRegion);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins
      "BuilderCallback":$bodyBuilder,
      "BuilderCallback":$condBuilder
    )>
  ];

  let assemblyFormat = [{
    $bodyRegion `while` $condRegion attr-dict
  }];
}

def HighLevel_BreakOp : ControlFlowOp< "break", [SoftTerminatorTrait] >
{
  let summary = "VAST break statement";
  let description = [{ VAST break statement }];

  let assemblyFormat = [{ attr-dict }];
}

def HighLevel_ContinueOp : ControlFlowOp< "continue", [SoftTerminatorTrait] >
{
  let summary = "VAST continue statement";
  let description = [{ VAST continue statement }];

  let assemblyFormat = [{ attr-dict }];
}

def HighLevel_SwitchOp : ControlFlowOp< "switch" >
{
  let summary = "VAST switch statement";
  let description = [{
    The operation represents a switch statement.

    The generic form of the operation is as follows:

    hl.switch {
      ... /* cond region */
      hl.value.yield %val : !hl.type
    } cases {
      ... /* casesregion */
    }
  }];

  let regions = (region ValueRegion:$condRegion, VariadicRegion<AnyRegion>:$cases);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins
      "BuilderCallback":$condBuilder,
      "BuilderCallback":$casesBuilder
    )>
  ];

  let assemblyFormat = [{
    $condRegion `cases` $cases attr-dict
  }];
}

def HighLevel_CaseOp : ControlFlowOp< "case" >
{
  let summary = "VAST case statement";
  let description = [{
    The operation represents a single case of a switch statement.

    The generic form of the operation is as follows:

    hl.case {
      ... /* lhs/check region */
      hl.value.yield %val : !hl.type
    } {
      ... /* body region */
    }

    It represents a C statement of form `case lhs: body;`.
  }];

  let regions = (region ValueRegion:$lhs, SizedRegion<1>:$body);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins
      "BuilderCallback":$lhsBuilder,
      "BuilderCallback":$bodyBuilder
    )>
  ];

  let assemblyFormat = [{ $lhs $body attr-dict }];
}

def HighLevel_DefaultOp : ControlFlowOp< "default" >
{
  let summary = "VAST default statement";
  let description = [{ VAST default statement }];

  let regions = (region SizedRegion<1>:$body);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins
      "BuilderCallback":$bodyBuilder
    )>
  ];

  let assemblyFormat = [{ $body attr-dict }];
}

def HighLevel_LabelDeclOp
  : HighLevel_Op< "label.decl", [] >
  , Arguments<(ins StrAttr:$name)>
  , Results<(outs LabelType:$result)>
{
  let assemblyFormat = [{ $name attr-dict `:` type($result) }];
}

def HighLevel_LabelStmt
  : ControlFlowOp< "label" >
  , Arguments<(ins LabelType:$label)>
{
  let regions = (region AnyRegion:$body);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins
      "mlir::Value":$label,
      "BuilderCallback":$substmtBuilder
    )>
  ];

  let assemblyFormat = [{ $label $body attr-dict }];
}

def HighLevel_GotoStmt
  : HighLevel_Op< "goto", [] >
  , Arguments<(ins LabelType:$label)>
{
  let assemblyFormat = [{ $label attr-dict }];
}

def HighLevel_SkipStmt : HighLevel_Op< "skip", [] >
{
  let summary = "VAST skip statement";
  let description = [{ VAST skip statement }];

  let assemblyFormat = [{ attr-dict }];
}

#endif // VAST_DIALECT_HIGHLEVEL_IR_HIGHLEVELCF
