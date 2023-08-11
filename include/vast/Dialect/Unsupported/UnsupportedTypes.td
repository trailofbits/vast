// Copyright (c) 2023-present, Trail of Bits, Inc.

#ifndef VAST_DIALECT_IR_UNSUPPORTED_TYPES
#define VAST_DIALECT_IR_UNSUPPORTED_TYPES

include "mlir/IR/SubElementInterfaces.td"

def UnsupportedType : UnsupportedType< "Unsupported", "type", [SubElementTypeInterface] > {
  let parameters = (ins StringRefParameter<>:$originName);

  let assemblyFormat = "`<` $originName `>`";
}

#endif // VAST_DIALECT_IR_UNSUPPORTED_TYPES
