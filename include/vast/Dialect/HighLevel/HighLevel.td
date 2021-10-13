// Copyright (c) 2021-present, Trail of Bits, Inc.

#ifndef VAST_DIALECT_HIGHLEVEL_IR_HIGHLEVEL
#define VAST_DIALECT_HIGHLEVEL_IR_HIGHLEVEL

include "vast/Dialect/HighLevel/HighLevelAttributes.td"
include "vast/Dialect/HighLevel/HighLevelTypes.td"

include "mlir/Interfaces/SideEffectInterfaces.td"

class HighLevel_Op< string mnemonic, list< OpTrait > traits = [] >
    : Op< HighLevel_Dialect, mnemonic, traits >;

class HighLevel_BinOp< string mnemonic, list< OpTrait > traits = [] >
    : HighLevel_Op< mnemonic, !listconcat(traits, [NoSideEffect, SameOperandsAndResultType]) >
{
    let summary = "VAST binary operation";
    let description = [{ VAST binary operation }];

    let arguments = (ins AnyType:$lhs, AnyType:$rhs);
    let results = (outs AnyType:$res);

    let verifier = ? ;

    let printer = [{
        p << getOperationName();
        p << " " << lhs() << ", " << rhs();
        p << " : " << getResult().getType();
    }];
}


class HighLevel_UnInplaceOp< string mnemonic, list< OpTrait > traits = [] >
    : HighLevel_Op< mnemonic, !listconcat(traits, []) >
{
    let summary = "VAST unary inplace operation";
    let description = [{ VAST unary inplace operation }];

    let arguments = (ins AnyType:$arg);

    let verifier = ? ;

    let printer = [{
        p << getOperationName() << " " << arg() << " : " << arg().getType();
    }];
}

class HighLevel_UnOp< string mnemonic, list< OpTrait > traits = [] >
    : HighLevel_Op< mnemonic, !listconcat(traits, [SameOperandsAndResultType]) >
{
    let summary = "VAST unary inplace operation";
    let description = [{ VAST unary inplace operation }];

    let arguments = (ins AnyType:$arg);
    let results = (outs AnyType:$res);

    let verifier = ? ;

    let printer = [{
        p << getOperationName() << " " << arg() << " : " << arg().getType();
    }];
}


class HighLevel_CompoundAssignOp< string mnemonic, list< OpTrait > traits = [] >
    : HighLevel_Op< mnemonic, !listconcat(traits, [SameTypeOperands]) >
{
    let summary = "VAST compound assign operation";
    let description = [{ VAST compound assign operation }];

    let arguments = (ins AnyType:$dst, AnyType:$src);

    let verifier = ? ;

    let printer = [{
        p << getOperationName() << " " << src() << " → " << dst();
    }];
}

include "HighLevelOps.td"

#endif // VAST_DIALECT_HIGHLEVEL_IR_HIGHLEVEL
