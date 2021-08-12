// Copyright (c) 2021-present, Trail of Bits, Inc.

#ifndef VAST_DIALECT_HIGHLEVEL_IR_HIGHLEVELTYPES
#define VAST_DIALECT_HIGHLEVEL_IR_HIGHLEVELTYPES

include "vast/Dialect/HighLevel/HighLevelBase.td"

class HighLevel_Type< string name, string type_mnemonic >
    : TypeDef< HighLevel_Dialect, name >
{
    let mnemonic = type_mnemonic;
}

// Builtin C Types

def Void : HighLevel_Type< "Void", "void" >
{
    let summary = "An high level void type representation.";
    let description = [{ void type }];
}

def Integer : HighLevel_Type< "Integer", "integer" >
{
    let summary = "An high level generic integer type representation.";
    let description = [{ integer type }];

    let parameters = (ins
        "integer_qualifier":$qualifier,
        "integer_kind":$kind
    );
}

#endif // VAST_DIALECT_HIGHLEVEL_IR_HIGHLEVELTYPES