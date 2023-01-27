// Copyright (c) 2022-present, Trail of Bits, Inc.

#ifndef VAST_DIALECT_CORE
#define VAST_DIALECT_CORE

include "vast/Interfaces/SymbolInterface.td"

def Core_Dialect : Dialect {
    let name = "core";
    let cppNamespace = "::vast::core";

    let summary = "Utility dialect to provide common features for other dialects.";

    let description = [{
        Dialect providing features that may be used by other dialects.
        These features can be used by including "vast/Dialect/Core/Core.td"
        It also provides lazy.op for lazy evaluation of expressions and
        binary logical operations that make use of it.
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

def ValueRegion : Region< HasOneBlock, "value region" >;

include "vast/Dialect/Core/CoreLazy.td"

#endif // VAST_DIALECT_CORE
