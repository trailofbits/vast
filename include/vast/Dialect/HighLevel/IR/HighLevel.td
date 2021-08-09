// Copyright (c) 2021-present, Trail of Bits, Inc.

#ifndef VAST_DIALECT_HIGHLEVEL_IR_HIGHLEVEL
#define VAST_DIALECT_HIGHLEVEL_IR_HIGHLEVEL

include "vast/Dialect/HighLevel/IR/HighLevelAttributes.td"
include "vast/Dialect/HighLevel/IR/HighLevelTypes.td"

include "mlir/Interfaces/SideEffectInterfaces.td"

class HighLevel_Op< string mnemonic, list< OpTrait > traits = [] >
    : Op< HighLevel_Dialect, mnemonic, traits >
{
    let verifier = "return ::verify(*this)";
}

#endif // VAST_DIALECT_HIGHLEVEL_IR_HIGHLEVEL
