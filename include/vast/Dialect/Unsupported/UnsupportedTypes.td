// Copyright (c) 2023-present, Trail of Bits, Inc.

#ifndef VAST_DIALECT_IR_UNSUPPORTED_TYPES
#define VAST_DIALECT_IR_UNSUPPORTED_TYPES

def UnsupportedType
    : Unsupported_Type< "Unsupported", "unsupported.type" > {
  let parameters = (ins OptionalParameter<"Type">:$elementType);
  let assemblyFormat = "`<` ( $elementType^ )? `>`";
}

#endif // VAST_DIALECT_IR_UNSUPPORTED_TYPES
