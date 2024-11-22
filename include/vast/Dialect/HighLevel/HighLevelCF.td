// Copyright (c) 2021-present, Trail of Bits, Inc.

#ifndef VAST_DIALECT_HIGHLEVEL_IR_HIGHLEVELCF
#define VAST_DIALECT_HIGHLEVEL_IR_HIGHLEVELCF

include "mlir/Interfaces/ControlFlowInterfaces.td"

include "vast/Dialect/Core/CoreTraits.td"
include "vast/Dialect/Core/Interfaces/SymbolInterface.td"

class HighLevel_ControlFlowOp< string mnemonic, list< Trait > traits = [] >
    : HighLevel_Op< mnemonic, !listconcat(traits,
        [NoTerminator, NoRegionArguments]
      ) >
{
    let summary = "VAST control flow operation";
    let description = [{ VAST control flow operation }];
}

def HighLevel_CondYieldOp
  : HighLevel_Op< "cond.yield", [
    // TODO(Heno): add ReturnLike trait
    Terminator, ParentOneOf<["IfOp", "WhileOp", "ForOp", "DoOp", "CondOp", "ChooseExprOp", "BinaryCondOp"]>
  ] >
{
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

def HighLevel_ValueYieldOp
  : HighLevel_Op< "value.yield", [
    // TODO(Heno): add ReturnLike trait
    Terminator
  ] >
{
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

def HighLevel_TypeYieldOp
  : HighLevel_Op< "type.yield", [
    // TODO(Heno): add ReturnLike trait
    Terminator
  ] >
{
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

def HighLevel_IfOp : HighLevel_ControlFlowOp< "if", [
  NoTerminator, Core_ScopeLikeTrait,
  Core_ShadowingSymbolTable< [
    [Core_VarSymbol, Core_TypeSymbol, Core_EnumConstantSymbol],
    [Core_ElaboratedTypeSymbol]
  ] >
] > {
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

  let regions = (region
    Core_CondRegion:$condRegion,
    AnyRegion:$thenRegion,
    AnyRegion:$elseRegion
  );

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins
      "builder_callback_ref":$condBuilder,
      "builder_callback_ref":$thenBuilder,
      CArg< "maybe_builder_callback_ref", "std::nullopt" >:$elseBuilder
    )>
  ];

  let extraClassDeclaration = [{
    /// Returns true if an else block exists.
    bool hasElse() { return !getElseRegion().empty(); }
  }];

  let hasCustomAssemblyFormat = 1;
}

class HighLevel_CondOpBase< string mnemonic, list< Trait > traits = [] >

  : HighLevel_ControlFlowOp< mnemonic, traits >
  , Results<(outs AnyType:$result)>
{
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
    Core_CondRegion:$condRegion,
    Core_ValueRegion:$thenRegion,
    Core_ValueRegion:$elseRegion
  );

  let hasRegionVerifier = 1;

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins
      "Type":$type,
      "builder_callback_ref":$condBuilder,
      "builder_callback_ref":$thenBuilder,
      "builder_callback_ref":$elseBuilder
    ), [{
        InsertionGuard guard($_builder);
        build_region($_builder, $_state, condBuilder);
        build_region($_builder, $_state, thenBuilder);
        build_region($_builder, $_state, elseBuilder);
        $_state.addTypes(type);
    }] >
  ];

  let assemblyFormat = [{ attr-dict `:` type(results) $condRegion `?` $thenRegion `:` $elseRegion }];
}

def HighLevel_CondOp       : HighLevel_CondOpBase< "cond" > {
  let summary = "VAST conditional statement";
}

def HighLevel_ChooseExprOp : HighLevel_ControlFlowOp< "choose_expr" >
  , Arguments<(ins OptionalAttr< BoolAttr >:$condTrue)>
  , Results<(outs AnyType:$result)>
{
  let summary = "Representation of GNU __builtin_choose_expr";

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
    Core_CondRegion:$condRegion,
    Core_ValueRegion:$thenRegion,
    Core_ValueRegion:$elseRegion
  );

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins
      "Type":$type,
      "builder_callback_ref":$condBuilder,
      "builder_callback_ref":$thenBuilder,
      "builder_callback_ref":$elseBuilder,
      CArg<"std::optional< bool >", "std::nullopt">:$condTrue
    ), [{
        InsertionGuard guard($_builder);
        build_region($_builder, $_state, condBuilder);
        build_region($_builder, $_state, thenBuilder);
        build_region($_builder, $_state, elseBuilder);
        if (condTrue) {
            $_state.addAttribute("condTrue", $_builder.getBoolAttr(condTrue.value()));
        }
        $_state.addTypes(type);
    }] >
  ];

  let extraClassDeclaration = [{
    std::optional< bool > isConditionTrue() {
        if (auto cond = getCondTrue()) {
            return cond.value();
        }
        return {};
    }

    bool isConditionDependent() { return (bool) getCondTrueAttr(); }

    region_ptr getChosenSubExpr() {
        if (auto cond = isConditionTrue()) {
            return cond.value() ? &getThenRegion() : &getElseRegion();
        }
        return {};
    }
  }];

  let assemblyFormat = [{ attr-dict (`cond` $condTrue^)? `:` type(results) $condRegion `?` $thenRegion `:` $elseRegion }];
}

def HighLevel_BinaryCondOp
  : HighLevel_Op<
        "binary_cond",
        [NoTerminator]
    >
    , Results<(outs AnyType:$result)>
{
  let summary = "VAST binary conditional statement";
  let description = [{
    Binary Conditional operator defined by the GNU extension of C.
    This operation holds in total 4 regions:
    1) The `common` region holds the common subexpression of the condition and result
    2) The `cond` region takes an argument, which should be the result of `common`,
       and yields the condition result.
    2) The `then` region takes an argument, which should be the result of `common`,
       and yields the result (it may potentially contain implicit casts).
    3) The `else` region contains the right hand side expression and yields the result.
  }];

  let regions = (region
    Core_ValueRegion:$commonRegion,
    Core_CondRegion:$condRegion,
    Core_ValueRegion:$thenRegion,
    Core_ValueRegion:$elseRegion
  );

  let hasRegionVerifier = 1;

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins
      "Type":$type,
      "builder_callback_ref":$commonBuilder,
      "builder_callback_ref":$condBuilder,
      "builder_callback_ref":$thenBuilder,
      "builder_callback_ref":$elseBuilder
    )>
  ];

  let assemblyFormat = [{ attr-dict `:` type(results) $commonRegion `,` $condRegion `?` $thenRegion `:` $elseRegion }];
}

def HighLevel_WhileOp : HighLevel_ControlFlowOp< "while", [
  NoTerminator, Core_ScopeLikeTrait,
  Core_ShadowingSymbolTable< [
    [Core_VarSymbol, Core_TypeSymbol, Core_EnumConstantSymbol],
    [Core_ElaboratedTypeSymbol]
  ] >
] > {
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

  let regions = (region
    Core_CondRegion:$condRegion,
    AnyRegion:$bodyRegion
  );

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins
      "builder_callback_ref":$condBuilder,
      "builder_callback_ref":$bodyBuilder
    )>
  ];

  let assemblyFormat = [{ $condRegion `do` $bodyRegion attr-dict }];
}


def HighLevel_ForOp : HighLevel_ControlFlowOp< "for", [
  NoTerminator, Core_ScopeLikeTrait,
  Core_ShadowingSymbolTable< [
    [Core_VarSymbol, Core_TypeSymbol, Core_EnumConstantSymbol],
    [Core_ElaboratedTypeSymbol]
  ] >
] > {
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
    Core_CondRegion:$condRegion,
    AnyRegion:$incrRegion,
    AnyRegion:$bodyRegion
  );

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins
      "builder_callback_ref":$condBuilder,
      "builder_callback_ref":$incrBuilder,
      "builder_callback_ref":$bodyBuilder
    )>
  ];

  let assemblyFormat = [{
    $condRegion `incr` $incrRegion attr-dict `do` $bodyRegion
  }];
}

def HighLevel_DoOp : HighLevel_ControlFlowOp< "do", [
  NoTerminator, Core_ScopeLikeTrait,
  // FIXME: cond region should not expose symbols to the body region
  Core_ShadowingSymbolTable< [
    [Core_VarSymbol, Core_TypeSymbol, Core_EnumConstantSymbol],
    [Core_ElaboratedTypeSymbol]
  ] >
] > {
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

  let regions = (region
    AnyRegion:$bodyRegion,
    Core_CondRegion:$condRegion
  );

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins
      "builder_callback_ref":$bodyBuilder,
      "builder_callback_ref":$condBuilder
    )>
  ];

  let assemblyFormat = [{
    $bodyRegion `while` $condRegion attr-dict
  }];
}

def HighLevel_BreakOp
  : HighLevel_ControlFlowOp< "break", [Core_SoftTerminatorTrait] >
{
  let summary = "VAST break statement";
  let description = [{ VAST break statement }];

  let assemblyFormat = [{ attr-dict }];
}

def HighLevel_ContinueOp
  : HighLevel_ControlFlowOp< "continue", [Core_SoftTerminatorTrait] >
{
  let summary = "VAST continue statement";
  let description = [{ VAST continue statement }];

  let assemblyFormat = [{ attr-dict }];
}

def HighLevel_SwitchOp : HighLevel_ControlFlowOp< "switch", [
  NoTerminator, Core_ScopeLikeTrait,
  // FIXME: look at proper scoping
  Core_ShadowingSymbolTable< [
    [Core_VarSymbol, Core_TypeSymbol, Core_EnumConstantSymbol],
    [Core_ElaboratedTypeSymbol]
  ] >
] > {
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

  let regions = (region
    Core_ValueRegion:$condRegion,
    VariadicRegion<AnyRegion>:$cases
  );

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins
      "builder_callback_ref":$condBuilder,
      "builder_callback_ref":$casesBuilder
    )>
  ];

  let assemblyFormat = [{
    $condRegion `cases` $cases attr-dict
  }];
}

def HighLevel_CaseOp
  : HighLevel_ControlFlowOp< "case" >
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

  let regions = (region
    Core_ValueRegion:$lhs,
    SizedRegion<1>:$body
  );

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins
      "builder_callback_ref":$lhsBuilder,
      "builder_callback_ref":$bodyBuilder
    )>
  ];

  let assemblyFormat = [{ $lhs $body attr-dict }];
}

def HighLevel_DefaultOp
  : HighLevel_ControlFlowOp< "default" >
{
  let summary = "VAST default statement";
  let description = [{ VAST default statement }];

  let regions = (region SizedRegion<1>:$body);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins
      "builder_callback_ref":$bodyBuilder
    )>
  ];

  let assemblyFormat = [{ $body attr-dict }];
}

def HighLevel_LabelDeclOp
  : HighLevel_Op< "label.decl", [Core_LabelSymbol] >
  , Arguments<(ins SymbolNameAttr:$sym_name)>
  , Results<(outs HighLevel_LabelType:$result)>
{
  let assemblyFormat = [{ $sym_name attr-dict `:` type($result) }];
}

def HighLevel_LabelStmt
  : HighLevel_ControlFlowOp< "label" >
  , Arguments<(ins HighLevel_LabelType:$label)>
{
  let regions = (region AnyRegion:$body);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins
      "mlir::Value":$label,
      "builder_callback_ref":$substmtBuilder
    )>
  ];

  let assemblyFormat = [{ $label $body attr-dict }];
}

def HighLevel_GotoStmt
  : HighLevel_Op< "goto", [] >
  , Arguments<(ins HighLevel_LabelType:$label)>
{
  let assemblyFormat = [{ $label attr-dict }];
}

def HighLevel_IndirectGotoStmt
  : HighLevel_Op< "indirect_goto", [] >
{
  let regions = (region
    Core_ValueRegion:$target
  );

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins
      "builder_callback_ref":$targetBuilder
    )>
  ];

  let assemblyFormat = [{ attr-dict `:` $target }];
}

def HighLevel_NullStmt : HighLevel_Op< "null", [] >
{
  let summary = "VAST null statement";
  let description = [{ VAST null statement }];

  let assemblyFormat = [{ attr-dict }];
}

#endif // VAST_DIALECT_HIGHLEVEL_IR_HIGHLEVELCF
