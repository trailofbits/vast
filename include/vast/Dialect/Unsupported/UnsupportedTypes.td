// Copyright (c) 2023-present, Trail of Bits, Inc.

#ifndef VAST_DIALECT_IR_UNSUPPORTED_TYPES
#define VAST_DIALECT_IR_UNSUPPORTED_TYPES

def UnsupportedType : UnsupportedType< "Unsupported", "type" > {
  let parameters = (ins StringRefParameter<>:$originName);

  let assemblyFormat = "`<` $originName `>`";
}

#endif // VAST_DIALECT_IR_UNSUPPORTED_TYPES
