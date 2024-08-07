// Copyright (c) 2024, Trail of Bits, Inc.

#pragma once

#include "vast/Interfaces/SymbolInterface.hpp"

#include "vast/Dialect/Core/CoreTraits.hpp"
#include "vast/Dialect/Core/SymbolTable.hpp"

namespace vast::core {

    template< typename concrete_type >
    struct ShadowingSymbolTable : op_trait_base< concrete_type, ShadowingSymbolTable > {};

} // namespace vast::core

/// Include the generated interface declarations.
#include "vast/Interfaces/SymbolTableInterface.h.inc"

namespace vast::core {

    using symbol_table_op_interface = SymbolTableOpInterface;

} // namespace vast::core
