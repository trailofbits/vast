// Copyright (c) 2022-present, Trail of Bits, Inc.

#ifndef VAST_DIALECT_CORE
#define VAST_DIALECT_CORE

include "mlir/IR/RegionKindInterface.td"

include "vast/Interfaces/SymbolInterface.td"

def Core_Dialect : Dialect {
    let name = "core";
    let cppNamespace = "::vast::core";

    let summary = "Utility dialect to provide common features for other dialects.";

    let description = [{
        ...
    }];

    let extraClassDeclaration = [{
        void registerTypes();
        void registerAttributes();
    }];

    let hasConstantMaterializer = 1;

    let emitAccessorPrefix = kEmitAccessorPrefix_Prefixed;
}

class Core_Op< string mnemonic, list< Trait > traits = [] >
    : Op< Core_Dialect, mnemonic, traits >;

def HasOneBlock   : CPred<"$_self.hasOneBlock()">;

include "CoreLazy.td"

#endif // VAST_DIALECT_CORE
