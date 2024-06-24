// Copyright (c) 2024, Trail of Bits, Inc.

#pragma once

#include "vast/Interfaces/SymbolInterface.hpp"

#include "vast/Dialect/Core/CoreTraits.hpp"

namespace vast::core {

    template< typename concrete_type >
    struct ShadowingSymbolTable : op_trait_base< concrete_type, ShadowingSymbolTable > {};

    template< typename symbols_list >
    bool holds_symbol_kind(symbol_kind kind) {
        if constexpr ( symbols_list::empty ) {
            return false;
        } else if (is_symbol_kind< typename symbols_list::head >(kind)) {
            return true;
        } else {
            return holds_symbol_kind< typename symbols_list::tail >(kind);
        }
    }

} // namespace vast::core

/// Include the generated interface declarations.
#include "vast/Interfaces/SymbolTableInterface.h.inc"
