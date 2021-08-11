// Copyright (c) 2021-present, Trail of Bits, Inc.

#ifndef VAST_DIALECT_HIGHLEVEL_IR_HIGHLEVELTYPES
#define VAST_DIALECT_HIGHLEVEL_IR_HIGHLEVELTYPES

include "vast/Dialect/HighLevel/HighLevelBase.td"

class HighLevel_Type< string name, string type_mnemonic >
    : TypeDef< HighLevel_Dialect, name >
{
    let mnemonic = type_mnemonic;
}

#endif // VAST_DIALECT_HIGHLEVEL_IR_HIGHLEVELTYPES