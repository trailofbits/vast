// Copyright (c) 2024, Trail of Bits, Inc.

#ifndef VAST_DIALECT_PARSER_OPS
#define VAST_DIALECT_PARSER_OPS

include "mlir/IR/OpBase.td"

include "vast/Dialect/Core/Utils.td"

class Parser_Op< string mnemonic, list< Trait > traits = [] >
  : Op< Parser_Dialect, mnemonic, traits >;

#endif // VAST_DIALECT_PARSER_OPS
