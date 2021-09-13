// Copyright (c) 2021-present, Trail of Bits, Inc.

#ifndef VAST_DIALECT_HIGHLEVEL_IR_HIGHLEVEL
#define VAST_DIALECT_HIGHLEVEL_IR_HIGHLEVEL

include "vast/Dialect/HighLevel/HighLevelAttributes.td"
include "vast/Dialect/HighLevel/HighLevelTypes.td"

include "mlir/Interfaces/SideEffectInterfaces.td"

class HighLevel_Op< string mnemonic, list< OpTrait > traits = [] >
    : Op< HighLevel_Dialect, mnemonic, traits >
{
    // let verifier = "return ::verify(*this)";
}

class HighLevel_BinOp< string mnemonic, list< OpTrait > traits = [] >
    : HighLevel_Op< mnemonic, traits >
{
    let summary = "VAST binary operation";
    let description = [{ VAST binary operation }];

    let arguments = (ins AnyType:$lhs, AnyType:$rhs);
    let results = (outs AnyType:$res);

    let builders = [
        OpBuilderDAG< (ins "mlir::Value":$lhs, "mlir::Value":$rhs),
        [{
            assert(lhs.getType() == rhs.getType());
            build($_builder, $_state, lhs.getType(), lhs, rhs);
        }] >
    ];

    let verifier = ? ;

    let printer = [{
        p << getOperationName();
        p << " " << lhs() << ", " << rhs();
        p << " : " << getResult().getType();
    }];
}

include "HighLevelOps.td"

#endif // VAST_DIALECT_HIGHLEVEL_IR_HIGHLEVEL
