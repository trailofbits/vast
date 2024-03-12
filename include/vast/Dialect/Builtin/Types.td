// Copyright (c) 2023, Trail of Bits, Inc.

#ifndef VAST_DIALECT_BUILTIN_TYPES
#define VAST_DIALECT_BUILTIN_TYPES

class HLBuiltin_Type< string name, string _mnemonic, list<Trait> traits = [] >
    : TypeDef< HLBuiltin_Dialect, name, traits >
{
    let mnemonic = _mnemonic;
}

def HLBuiltin_VAListType : HLBuiltin_Type<"VAList", "va_list"> {
    let summary = "va_list type";

    let description = [{
        Represents target-agnostic va_list type.
    }];
}

#endif // VAST_DIALECT_BUILTIN_TYPES
