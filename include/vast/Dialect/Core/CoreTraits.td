// Copyright (c) 2022-present, Trail of Bits, Inc.

#ifndef VAST_DIALECT_CORE_CORELAZY
#define VAST_DIALECT_CORE_CORELAZY

include "mlir/IR/DialectBase.td"

def soft_terminator : NativeOpTrait< "soft_terminator" >, StructuralOpTrait
{
    let cppNamespace = "::vast::core";
}

#endif //VAST_DIALECT_CORE_CORELAZY
