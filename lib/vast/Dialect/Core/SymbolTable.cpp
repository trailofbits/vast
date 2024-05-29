// Copyright (c) 2024, Trail of Bits, Inc.

#include "vast/Dialect/Core/SymbolTable.hpp"

namespace vast::core {

    string_ref symbol_attr_name() {
        return mlir::SymbolTable::getSymbolAttrName();
    }

} // namespace vast::core
