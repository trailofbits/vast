// Copyright (c) 2021-present, Trail of Bits, Inc.

#ifndef VAST_DIALECT_IR_LOWLEVELOPS
#define VAST_DIALECT_IR_LOWLEVELOPS

include "mlir/IR/BuiltinAttributes.td"

include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "vast/Interfaces/SymbolInterface.td"

// TODO(lukas): Add type constraints.
def LowLevel_StructGEPOp
  : LowLevel_Op< "gep", [] >
  , Arguments<(ins AnyType:$record, I32Attr:$idx, StrAttr:$name)>
  , Results<(outs AnyType:$element)>
{
    let summary = "VAST struct gep  operation";
    let description = [{ VAST struct gep operation }];
}

def UninitializedVar
    : LowLevel_Op< "unintialized_var", [VastSymbol] >
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
    : LowLevel_Op< "br",
        [Terminator, DeclareOpInterfaceMethods<BranchOpInterface>]
      >
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
    , Arguments<(ins AnyType:$cond, Variadic<AnyType>:$trueOperands, Variadic<AnyType>:$falseOperands )>
{
    let summary = "Conditional branch.";
    let description = [{ Direct branch }];

    let successors = (successor AnySuccessor:$trueDest, AnySuccessor:$falseDest);

    let builders = [
        OpBuilder< (ins
            "mlir::Value":$cond,
            "mlir::ValueRange":$trueOperands, "mlir::Block *":$trueDest,
            "mlir::ValueRange":$falseOperands, "mlir::Block *":$falseDest),
        [{
            build($_builder, $_state,
                  cond,
                  trueOperands, falseOperands,
                  trueDest, falseDest);
        }] >,
        OpBuilder< (ins
            "mlir::Value":$cond,
            "mlir::Block *":$trueDest,
            "mlir::Block *":$falseDest),
        [{
            build($_builder, $_state,
                  cond,
                  mlir::ValueRange(), mlir::ValueRange(),
                  trueDest, falseDest);
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
    let arguments = (ins AnyType:$cond, Variadic<AnyType>:$cond_operands);

    let builders = [

        OpBuilder< (ins "mlir::Value":$cond, "mlir::Block *":$dest),
        [{
            build($_builder, $_state, cond, mlir::ValueRange(), dest);
        }] >
    ];
}

def ReturnOp
    : LowLevel_Op< "return", [Terminator] >
    , Arguments<(ins Variadic<AnyType>:$result)>
{
    let assemblyFormat = "($result^ `:` type($result))? attr-dict";

    // Allow building a ReturnOp with no return operand.
    let builders = [
        OpBuilder<(ins), [{ build($_builder, $_state, std::nullopt); }]>
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
            auto size = std::distance(getBody().begin(), getBody().end());
            if (size < 2)
                return nullptr;
            return &*std::next(getBody().begin());
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
