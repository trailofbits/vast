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
         , BuildableType< "VoidType::get($_builder.getContext())" >
{
    let summary = "An high level void type representation.";
    let description = [{ void type }];
}

def Bool : HighLevel_Type< "Bool", "bool" >
         , BuildableType< "BoolType::get($_builder.getContext())" >
{
    let summary = "An high level bool type representation.";
    let description = [{ bool type }];

    let parameters = (ins
        "bool":$isVolatile,
        "bool":$isConst
    );

    let builders = [
        TypeBuilder< (ins
            CArg<"bool", "false">:$isVolatile,
            CArg<"bool", "false">:$isConst ),
        [{
            return Base::get($_ctxt, isVolatile, isConst);
        }]>
    ];

    let skipDefaultBuilders = 1;

    let printer = [{
        if (getIsConst())
            $_printer << "const ";
        if (getIsVolatile())
            $_printer << "volatile ";
        $_printer << "bool";
    }];

    let parser = [{
        // TODO(Heno): parse qualifiers
        return get($_ctxt);
    }];
}

def Integer : HighLevel_Type< "Integer", "integer" >
{
    let summary = "An high level generic integer type representation.";
    let description = [{ integer type }];

    let parameters = (ins
        "signedness_qualifier":$signedness,
        "integer_kind":$kind,
        "bool":$isVolatile,
        "bool":$isConst
    );

    let builders = [
        TypeBuilder< (ins
            "signedness_qualifier":$signedness,
            "integer_kind":$kind,
            CArg<"bool", "false">:$isVolatile,
            CArg<"bool", "false">:$isConst ),
        [{
            return Base::get($_ctxt, signedness, kind, isVolatile, isConst);
        }]>
    ];

    let skipDefaultBuilders = 1;

    let printer = [{
        if (getIsConst())
            $_printer << "const ";
        if (getIsVolatile())
            $_printer << "volatile ";
        if (getSignedness() == vast::hl::signedness_qualifier::Unsigned)
            $_printer << "unsigned ";
        $_printer << to_string(getKind());
    }];

    let parser = [{
        auto sign = vast::hl::signedness_qualifier::Signed;

        // TODO(Heno): qualifier parser

        if ($_parser.parseOptionalKeyword("unsigned"))
            sign = vast::hl::signedness_qualifier::Unsigned;

        // TODO(Heno): integer kinds

        return get($_ctxt, sign, vast::hl::integer_kind::Int);
    }];
}

#endif // VAST_DIALECT_HIGHLEVEL_IR_HIGHLEVELTYPES
