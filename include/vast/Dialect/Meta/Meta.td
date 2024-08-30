// Copyright (c) 2021-present, Trail of Bits, Inc.

#ifndef VAST_DIALECT_META
#define VAST_DIALECT_META

include "mlir/IR/OpBase.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/RegionKindInterface.td"

include "vast/Dialect/Core/Interfaces/SymbolInterface.td"

def Meta_Dialect : Dialect {
    let name = "meta";
    let cppNamespace = "::vast::meta";

    let summary = "A vast metadata dialect.";

    let description = [{
        This dialect intends capture user metadata
        that are kept accross transformations.
    }];

    let extraClassDeclaration = [{
        void registerTypes();
        void registerAttributes();
    }];

    let useDefaultAttributePrinterParser = 1;
}

include "MetaAttributes.td"

#endif // VAST_DIALECT_META
