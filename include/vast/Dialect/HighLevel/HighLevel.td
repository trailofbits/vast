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
}

class HighLevel_Op< string mnemonic, list< OpTrait > traits = [] >
    : Op< HighLevel_Dialect, mnemonic, traits >;

class HighLevel_BinOp< string mnemonic, list< OpTrait > traits = [] >
    : HighLevel_Op< mnemonic, !listconcat(traits, [NoSideEffect, SameOperandsAndResultType]) >
{
    let summary = "VAST binary operation";
    let description = [{ VAST binary operation }];

    let arguments = (ins AnyType:$lhs, AnyType:$rhs);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{ $lhs `,` $rhs attr-dict `:` type($result) }];
}


class HighLevel_UnInplaceOp< string mnemonic, list< OpTrait > traits = [] >
    : HighLevel_Op< mnemonic, !listconcat(traits, []) >
{
    let summary = "VAST unary inplace operation";
    let description = [{ VAST unary inplace operation }];

    let arguments = (ins AnyType:$arg);

    let assemblyFormat = [{ $arg attr-dict `:` type($arg) }];
}

class HighLevel_TypePreservingUnOp< string mnemonic, list< OpTrait > traits = [] >
    : HighLevel_Op< mnemonic, !listconcat(traits, [NoSideEffect, SameOperandsAndResultType]) >
{
    let summary = "VAST unary type preserving operation";
    let description = [{ VAST unary type preserving operation }];

    let arguments = (ins AnyType:$arg);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{ $arg attr-dict `:` type($result) }];
}

class HighLevel_UnOp< string mnemonic, list< OpTrait > traits = [] >
    : HighLevel_Op< mnemonic, !listconcat(traits, [NoSideEffect]) >
{
    let summary = "VAST unary operation";
    let description = [{ VAST unary operation }];

    let arguments = (ins AnyType:$arg);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{ $arg attr-dict `:` type($arg) `->` type($result) }];
}


class HighLevel_CompoundAssignOp< string mnemonic, list< OpTrait > traits = [] >
    : HighLevel_Op< mnemonic, !listconcat(traits, [SameTypeOperands]) >
{
    let summary = "VAST compound assign operation";
    let description = [{ VAST compound assign operation }];

    let arguments = (ins AnyType:$dst, AnyType:$src);

    let assemblyFormat = [{ $src `to` $dst attr-dict `:` type($src) }];
}

include "HighLevelAttributes.td"
include "HighLevelTypes.td"
include "HighLevelOps.td"
include "HighLevelCF.td"

#endif // VAST_DIALECT_HIGHLEVEL_IR_HIGHLEVEL
