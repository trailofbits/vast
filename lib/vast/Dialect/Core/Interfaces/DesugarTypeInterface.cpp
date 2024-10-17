// Copyright (c) 2024-present, Trail of Bits, Inc.

#include "vast/Dialect/Core/Interfaces/DesugarTypeInterface.hpp"
#include "vast/Util/Common.hpp"

//===----------------------------------------------------------------------===//
// Desugar Type Interfaces
//===----------------------------------------------------------------------===//

namespace vast::core {
    mlir_type desugar_type(mlir_type type) {
        if (auto desugarable = mlir::dyn_cast< DesugarTypeInterface >(type))
            return desugarable.getDesugaredType();
        return type;
    }
} // namespace vast::core
/// Include the generated symbol interfaces.
#include "vast/Dialect/Core/Interfaces/DesugarTypeInterface.cpp.inc"
