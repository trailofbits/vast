// Copyright (c) 2021-present, Trail of Bits, Inc.

#ifndef VAST_DIALECT_HIGHLEVEL_IR_HIGHLEVEL
#define VAST_DIALECT_HIGHLEVEL_IR_HIGHLEVEL

include "mlir/IR/OpBase.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/RegionKindInterface.td"

def HighLevel_Dialect : Dialect {
    let name = "hl";
    let summary = "A high-level verbose program analysis MLIR dialect.";
    let description = [{
        This dialect intends capture highevel constructs of C/C++
        for further program analysis.
    }];
    let cppNamespace = "::vast::hl";
    let extraClassDeclaration = [{
        void registerTypes();
        void registerAttributes();
    }];

    let hasConstantMaterializer = 1;
}

class HighLevel_Op< string mnemonic, list< OpTrait > traits = [] >
    : Op< HighLevel_Dialect, mnemonic, traits >;

class HighLevel_BinOp< string mnemonic, list< OpTrait > traits = [] >
    : HighLevel_Op< mnemonic, !listconcat(traits, [NoSideEffect, SameOperandsAndResultType]) >
{
    let summary = "VAST binary operation";
    let description = [{
        High-level binary operation. This operation takes two operands
        and returns one result, each of these is required to be of the same
        type.

        The custom assembly form of the operation is as follows:

        %result = <op> %lhs, %rhs  : type
    }];

    let arguments = (ins AnyType:$lhs, AnyType:$rhs);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{ $lhs `,` $rhs attr-dict `:` type($result) }];
}

class HighLevel_UnInplaceOp< string mnemonic, list< OpTrait > traits = [] >
    : HighLevel_Op< mnemonic, !listconcat(traits, [SameOperandsAndResultType]) >
{
    let summary = "VAST unary inplace operation";
    let description = [{
        Inplace high-level unary operation changes its single argument in place.
        It does not produce a new value.

        The custom assembly form of the operation is as follows:

        %result = <op> %arg : type
    }];

    let arguments = (ins AnyType:$arg);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{ $arg attr-dict `:` type($result) }];
}

class HighLevel_TypePreservingUnOp< string mnemonic, list< OpTrait > traits = [] >
    : HighLevel_Op< mnemonic, !listconcat(traits, [NoSideEffect, SameOperandsAndResultType]) >
{
    let summary = "VAST unary type preserving operation";
    let description = [{
        Type preserving high-level unary operation assures that argument and
        result has the same type.

        The custom assembly form of the operation is as follows:

        %result = <op> %arg : type
    }];

    let arguments = (ins AnyType:$arg);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{ $arg attr-dict `:` type($result) }];
}

class HighLevel_UnOp< string mnemonic, list< OpTrait > traits = [] >
    : HighLevel_Op< mnemonic, !listconcat(traits, [NoSideEffect]) >
{
    let summary = "VAST unary operation";
    let description = [{
        A generic high-level unary operation can change the type of its argument.

        The custom assembly form of the operation is as follows:

        %result = <op> %arg  : type -> result_type
    }];

    let arguments = (ins AnyType:$arg);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{ $arg attr-dict `:` type($arg) `->` type($result) }];
}


def HasOneBlock : CPred<"$_self.hasOneBlock()">;

include "HighLevelAttributes.td"
include "HighLevelTypes.td"
include "HighLevelVar.td"
include "HighLevelOps.td"
include "HighLevelCF.td"

#endif // VAST_DIALECT_HIGHLEVEL_IR_HIGHLEVEL
