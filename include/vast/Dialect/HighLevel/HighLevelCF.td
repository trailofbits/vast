// Copyright (c) 2021-present, Trail of Bits, Inc.

#ifndef VAST_DIALECT_HIGHLEVEL_IR_HIGHLEVELCF
#define VAST_DIALECT_HIGHLEVEL_IR_HIGHLEVELCF

def HighLevel_CondYieldOp : HighLevel_Op< "cond.yield", [
  // TODO(HEno): add ReturnLike trait
  NoSideEffect, Terminator, ParentOneOf<["IfOp"]>
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

def HighLevel_IfOp : HighLevel_Op< "if", [SingleBlock, NoTerminator, NoRegionArguments, RecursiveSideEffects] >
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

#endif // VAST_DIALECT_HIGHLEVEL_IR_HIGHLEVELCF
