// Copyright (c) 2024, Trail of Bits, Inc.

#ifndef VAST_DIALECT_PARSER_OPS
#define VAST_DIALECT_PARSER_OPS

include "mlir/IR/OpBase.td"

include "vast/Dialect/Core/Utils.td"
include "vast/Dialect/Core/Interfaces/SymbolInterface.td"
include "vast/Dialect/Parser/Types.td"

class Parser_Op< string mnemonic, list< Trait > traits = [] >
  : Op< Parser_Dialect, mnemonic, traits >;

def Parser_Source
  : Parser_Op< "source" >
  , Arguments< (ins Variadic<Parser_AnyDataType>:$arguments) >
  , Results< (outs Variadic<Parser_AnyDataType>:$result) >
{
  let summary = "Source of parsed data.";

  let assemblyFormat = [{
    $arguments attr-dict `:` functional-type($arguments, $result)
  }];
}

def Paser_Sink
  : Parser_Op< "sink" >
  , Arguments< (ins Variadic<Parser_AnyDataType>:$arguments) >
  , Results< (outs Variadic<Parser_NoDataType>:$result) >
{
  let summary = "Sink of parsed data.";

  let assemblyFormat = [{
    $arguments attr-dict `:` functional-type($arguments, $result)
  }];
}

def Parser_Parse
  : Parser_Op< "parse" >
  , Arguments< (ins Variadic<Parser_AnyDataType>:$arguments) >
  , Results< (outs Variadic<Parser_AnyDataType>:$result) >
{
  let summary = "Parsing operation data.";

  let assemblyFormat = [{
    $arguments attr-dict `:` functional-type($arguments, $result)
  }];
}

def Parse_NoParse
  : Parser_Op< "noparse" >
  , Arguments< (ins Variadic<Parser_NoDataType>:$arguments) >
  , Results< (outs Variadic<Parser_NoDataType>:$result) >
{
  let summary = "Non-parsing operation data.";

  let hasFolder = 1;

  let assemblyFormat = [{
    $arguments attr-dict `:` functional-type($arguments, $result)
  }];
}

def Parse_ToMaybe
  : Parser_Op< "to_maybe" >
  , Arguments< (ins Parser_AnyDataType:$arguments) >
  , Results< (outs Parser_MaybeDataType:$result) >
{
  let summary = "Casting operation to maybe type.";

  let assemblyFormat = [{
    $arguments attr-dict `:` functional-type($arguments, $result)
  }];
}

def Parse_Decl
  : Parser_Op< "decl", [Core_VarSymbol] >
  , Arguments< (ins
    SymbolNameAttr:$sym_name,
    TypeAttr:$type
  ) >
{
  let summary = "Declaration operation.";

  let assemblyFormat = [{
    $sym_name attr-dict `:` $type
  }];
}

def Parse_Ref
  : Parser_Op< "ref" >
  , Arguments< (ins FlatSymbolRefAttr:$name) >
  , Results< (outs Parser_AnyDataType:$result) >
{
  let summary = "Parser variable reference operation.";

  let assemblyFormat = [{
    $name attr-dict `:` type($result)
  }];
}

#endif // VAST_DIALECT_PARSER_OPS
