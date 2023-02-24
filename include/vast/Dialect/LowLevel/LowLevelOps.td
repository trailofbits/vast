// Copyright (c) 2021-present, Trail of Bits, Inc.

#ifndef VAST_DIALECT_IR_LOWLEVELOPS
#define VAST_DIALECT_IR_LOWLEVELOPS

include "mlir/IR/BuiltinAttributes.td"

include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "vast/Interfaces/SymbolInterface.td"

// TODO(lukas): Add type constraints.
def LowLevel_StructGEPOp
  : LowLevel_Op< "gep", [NoSideEffect] >
  , Arguments<(ins AnyType:$record, I32Attr:$idx, StrAttr:$name)>
  , Results<(outs AnyType:$element)>
{
    let summary = "VAST struct gep  operation";
    let description = [{ VAST struct gep operation }];
}

def UninitializedVar
    : LowLevel_Op< "unintialized_var", [VastSymbol, NoSideEffect] >
    , Results<(outs AnyType:$result)>
{
    let summary = "Declaration of variable that have not been initialized yet.";
    let description = [{ Declaration of variable that have not been initialized yet. }];

    let results = (outs AnyType:$result);
}

def InitializeVar
    : LowLevel_Op< "initialize" >
    , Arguments<(ins AnyType:$var, Variadic<AnyType>:$elements)>
    , Results<(outs AnyType:$result)>
{
    let summary = "Initialize a variable.";
    let description = [{
        Initialize a variable - for now this operation is a direct lowering from hl.var
        initialization section. Later there will be need to discover how this ties
        to constructors.
    }];
}

def Br
    : LowLevel_Op< "br", [Terminator, NoSideEffect,
                          DeclareOpInterfaceMethods<BranchOpInterface>] >
{
    let summary = "Direct branch.";
    let description = [{ Direct branch }];

    let successors = (successor AnySuccessor:$dest);
    let arguments = (ins Variadic<AnyType>:$operands);

    let builders = [
        OpBuilder< (ins "mlir::ValueRange":$operands, "mlir::SuccessorRange":$dest),
        [{
            $_state.addOperands(operands);
            $_state.addSuccessors(dest);
        }] >,

        OpBuilder< (ins "mlir::Block *":$dest),
        [{
            $_state.addSuccessors(dest);
        }] >
    ];

    let assemblyFormat = [{
        $dest (`(` $operands^ `:` type($operands) `)`)? attr-dict
    }];
}

def CondBr
    : LowLevel_Op< "cond_br", [Terminator, AttrSizedOperandSegments] >
{
    let summary = "Conditional branch.";
    let description = [{ Direct branch }];

    let arguments = (ins AnyType:$cond,
                         Variadic<AnyType>:$true_operands,
                         Variadic<AnyType>:$false_operands);
    let successors = (successor AnySuccessor:$true_dest, AnySuccessor:$false_dest);

    let builders = [
        OpBuilder< (ins "mlir::Value":$cond,
                        "mlir::ValueRange":$true_operands, "mlir::Block *":$true_dest,
                        "mlir::ValueRange":$false_operands, "mlir::Block *":$false_dest),
        [{
            build($_builder, $_state,
                  cond,
                  true_operands, false_operands,
                  true_dest, false_dest);
        }] >,
        OpBuilder< (ins "mlir::Value":$cond,
                        "mlir::Block *":$true_dest, "mlir::Block *":$false_dest),
        [{
            build($_builder, $_state,
                  cond,
                  mlir::ValueRange(), mlir::ValueRange(),
                  true_dest, false_dest);
        }] >
    ];
}

def ScopeRet
    : LowLevel_Op< "scope_ret", [Terminator] >
{
    let summary = "Terminator of scope.";
    let description = [{ Terminator of scopes (for example during lowering of loops). }];
}

def ScopeRecurse
    : LowLevel_Op< "scope_recurse", [Terminator] >
{
    let summary = "Jump to first block of scope.";
    let description = [{ Modelling continue. }];
}

def CondScopeRet
    : LowLevel_Op< "cond_scope_ret", [Terminator] >
{
    let summary = "Terminator of scope if condition is met, otherwise branch.";
    let description = [{ Terminate or branch. }];

    let successors = (successor AnySuccessor:$dest);
    let arguments = (ins AnyType:$cond, Variadic<AnyType>:$operands);

    let builders = [

        OpBuilder< (ins "mlir::Value":$cond, "mlir::Block *":$dest),
        [{
            build($_builder, $_state, cond, mlir::ValueRange(), dest);
        }] >
    ];
}

def Scope
    : LowLevel_Op< "scope", [NoRegionArguments] >
{
    let summary = "Scope, holds one region.";
    let description = [{ Scope that holds one region, each block should be terminated
                         with either branch, scope return or their conditional variants. }];

    let regions = (region AnyRegion:$body);

    let extraClassDeclaration = [{
        mlir::Block *start_block()
        {
            auto size = std::distance(body().begin(), body().end());
            if (size < 2)
                return nullptr;
            return &*std::next(body().begin());
        }
    }];
}

def InlineScope
    : LowLevel_Op< "inline_scope", [NoRegionArguments] >
{
    let summary = "Scope, that forwards (cond)scope return up.";
    let description = [{ Result of inlined if. }];

    let regions = (region AnyRegion:$body);
}

#endif // VAST_DIALECT_IR_LOWLEVELOPS
