// Copyright (c) 2024, Trail of Bits, Inc.

#ifndef VAST_INTERFACES_SYMBOL_REF_INTERFACE
#define VAST_INTERFACES_SYMBOL_REF_INTERFACE

include "mlir/IR/OpBase.td"
include "vast/Dialect/Core/Interfaces/Common.td"

def Core_SymbolRefInterface : Core_AttrInterface< "SymbolRefInterface" > {
    let description = [{ This attribute represents a reference to a VAST symbol. }];
}

#endif // VAST_INTERFACES_SYMBOL_REF_INTERFACE
