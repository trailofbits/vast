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

def Bool : HighLevel_Type< "Bool", "bool" >
{
    let summary = "An high level bool type representation.";
    let description = [{ bool type }];
}

def Integer : HighLevel_Type< "Integer", "integer" >
{
    let summary = "An high level generic integer type representation.";
    let description = [{ integer type }];

    let parameters = (ins
        "integer_qualifier":$qualifier,
        "integer_kind":$kind
    );

    let printer = [{
        if (getQualifier() == vast::hl::integer_qualifier::Unsigned)
            $_printer << "unsigned ";
        $_printer << to_string(getKind());
    }];

    let parser = [{
        vast::hl::integer_qualifier qual = vast::hl::integer_qualifier::Signed;

        if ($_parser.parseOptionalKeyword("unsigned"))
            qual = vast::hl::integer_qualifier::Unsigned;
        // TODO(Heno): integer kinds
        return get($_ctxt, qual, vast::hl::integer_kind::Int);
    }];
}

#endif // VAST_DIALECT_HIGHLEVEL_IR_HIGHLEVELTYPES
