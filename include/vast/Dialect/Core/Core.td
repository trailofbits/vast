// Copyright (c) 2022-present, Trail of Bits, Inc.

#ifndef VAST_DIALECT_CORE
#define VAST_DIALECT_CORE

include "vast/Dialect/Core/Interfaces/SymbolInterface.td"

include "vast/Dialect/Core/Utils.td"

def Core_Dialect : Dialect {
    let name = "core";
    let cppNamespace = "::vast::core";

    let summary = "Utility dialect to provide common features for other dialects.";

    let description = [{
        Dialect providing features that may be used by other dialects.
        These features can be used by including "vast/Dialect/Core/Utils.td"
        It also provides lazy.op for lazy evaluation of expressions and
        binary logical operations that make use of it.
    }];

    let extraClassDeclaration = [{
        void registerTypes();
        void registerAttributes();

        static std::string getTargetTripleAttrName() { return "vast.core.target_triple"; }
        static std::string getLanguageAttrName() { return "vast.core.lang"; }
    }];

    let useDefaultTypePrinterParser = 1;
    let useDefaultAttributePrinterParser = 1;

    let hasConstantMaterializer = 1;
}

class Core_Op< string mnemonic, list< Trait > traits = [] >
    : Op< Core_Dialect, mnemonic, traits >;

include "vast/Dialect/Core/CoreLazy.td"
include "vast/Dialect/Core/CoreTypes.td"
include "vast/Dialect/Core/CoreOps.td"
include "vast/Dialect/Core/CoreAttributes.td"

#endif // VAST_DIALECT_CORE
