// Copyright (c) 2024, Trail of Bits, Inc.

#ifndef VAST_DIALECT_PARSER_TYPES
#define VAST_DIALECT_PARSER_TYPES

class Parser_Type< string name, string _mnemonic, list<Trait> traits = [] >
    : TypeDef< Parser_Dialect, name, traits >
{
    let mnemonic = _mnemonic;
}

#endif // VAST_DIALECT_PARSER_TYPES
