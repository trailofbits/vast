// Copyright (c) 2024, Trail of Bits, Inc.

#ifndef VAST_INTERFACES_SYMBOL_REF_INTERFACE
#define VAST_INTERFACES_SYMBOL_REF_INTERFACE

include "mlir/IR/OpBase.td"
include "vast/Interfaces/Common.td"

def VastSymbolRefInterface : VastAttrInterface< "VastSymbolRefInterface" > {
    let description = [{ This attribute represents a reference to a symbol. }];
}

#endif // VAST_INTERFACES_SYMBOL_REF_INTERFACE
