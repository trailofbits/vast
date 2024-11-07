// Copyright (c) 2024, Trail of Bits, Inc.

#ifndef VAST_DIALECT_PARSER_OPS
#define VAST_DIALECT_PARSER_OPS

include "mlir/IR/OpBase.td"

include "vast/Dialect/Core/Utils.td"
include "vast/Dialect/Parser/Types.td"

class Parser_Op< string mnemonic, list< Trait > traits = [] >
  : Op< Parser_Dialect, mnemonic, traits >;

def Parser_Source
  : Parser_Op< "source" >
  , Arguments< (ins Variadic<AnyType>:$arguments) >
  , Results< (outs Variadic<Parser_DataType>:$result) >
{
  let summary = "Source of parsed data.";
}

def Paser_Sink
  : Parser_Op< "sink" >
  , Arguments< (ins Variadic<Parser_MaybeDataType>:$arguments) >
  , Results< (outs Variadic<Parser_NoDataType>:$result) >
{
  let summary = "Sink of parsed data.";
}

def Parser_Parse
  : Parser_Op< "parse" >
  , Arguments< (ins Variadic<Parser_MaybeDataType>:$arguments) >
  , Results< (outs Variadic<AnyType>:$result) >
{
  let summary = "Parsing operation data.";
}

def Parse_NoParse
  : Parser_Op< "noparse" >
  , Arguments< (ins Variadic<AnyType>:$arguments) >
  , Results< (outs Variadic<AnyType>:$result) >
{
  let summary = "Non-parsing operation data.";

  let assemblyFormat = [{
    $arguments attr-dict `:` functional-type($arguments, $result)
  }];
}

#endif // VAST_DIALECT_PARSER_OPS
