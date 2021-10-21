// Copyright (c) 2021-present, Trail of Bits, Inc.

#ifndef VAST_DIALECT_HIGHLEVEL_IR_HIGHLEVELCF
#define VAST_DIALECT_HIGHLEVEL_IR_HIGHLEVELCF

class ControlFlowOp< string mnemonic, list< OpTrait > traits = [] >
    : HighLevel_Op< mnemonic, !listconcat(traits,
        [SingleBlock, NoTerminator, NoRegionArguments, RecursiveSideEffects]
      ) >
{
    let summary = "VAST control flow operation";
    let description = [{ VAST control flow operation }];
}

def HighLevel_CondYieldOp : HighLevel_Op< "cond.yield", [
  // TODO(Heno): add ReturnLike trait
  NoSideEffect, Terminator, ParentOneOf<["IfOp", "WhileOp", "ForOp", "DoOp"]>
] > {
  let summary = "condition yield operation";
  let description = [{ "condition yield operation" }];

  let arguments = (ins Bool:$result);
  let results = (outs);

  let assemblyFormat = [{ attr-dict $result `:` type($result) }];
}


def HasOneBlock : CPred<"$_self.hasOneBlock()">;

// TODO(Heno): constraints on cond region
def CondRegion : Region< HasOneBlock, "condition region" >;

def HighLevel_IfOp : ControlFlowOp< "if" >
{
  let summary = "VAST if statement";
  let description = [{ VAST if statement }];

  let regions = (region CondRegion:$condRegion, AnyRegion:$thenRegion, AnyRegion:$elseRegion);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins
      "BuilderCallback":$condBuilder,
      "BuilderCallback":$thenBuilder,
      CArg< "BuilderCallback", "nullptr">:$elseBuilder
    )>
  ];

  let extraClassDeclaration = [{
    /// Returns true if an else block exists.
    bool hasElse() { return !elseRegion().empty(); }
  }];

  let assemblyFormat = [{ $condRegion `then` $thenRegion (`else` $elseRegion^)? attr-dict }];
}


def HighLevel_WhileOp : ControlFlowOp< "while" >
{
  let summary = "VAST while statement";
  let description = [{ VAST while statement }];

  let regions = (region CondRegion:$condRegion, SizedRegion<1>:$bodyRegion);

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
  let description = [{ VAST for statement }];

  let regions = (region SizedRegion<1>:$initRegion, CondRegion:$condRegion,
                        SizedRegion<1>:$incrRegion, SizedRegion<1>:$bodyRegion);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins
      "BuilderCallback":$initBuilder,
      "BuilderCallback":$condBuilder,
      "BuilderCallback":$incrBuilder,
      "BuilderCallback":$bodyBuilder
    )>
  ];

  let assemblyFormat = [{
    $initRegion `cond` $condRegion `incr` $incrRegion attr-dict `do` $bodyRegion
  }];
}

def HighLevel_DoOp : ControlFlowOp< "do" >
{
  let summary = "VAST do-while statement";
  let description = [{ VAST do-while statement }];

  let regions = (region SizedRegion<1>:$bodyRegion, CondRegion:$condRegion);

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

def HighLevel_BreakOp : ControlFlowOp< "break" >
{
  let summary = "VAST break statement";
  let description = [{ VAST break statement }];

  let assemblyFormat = [{ attr-dict }];
}

def HighLevel_ContinueOp : ControlFlowOp< "continue" >
{
  let summary = "VAST continue statement";
  let description = [{ VAST continue statement }];

  let assemblyFormat = [{ attr-dict }];
}

#endif // VAST_DIALECT_HIGHLEVEL_IR_HIGHLEVELCF
