// Copyright (c) 2024, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <llvm/ADT/DenseMap.h>
#include <mlir/IR/OpDefinition.h>
VAST_UNRELAX_WARNINGS

#include "vast/Dialect/Core/CoreTraits.hpp"
#include "vast/Util/Common.hpp"

namespace vast::core {

    template< typename... SymbolInterfaces >
    struct ShadowingSymbolTable
    {
        template< typename ConcreteType >
        struct Impl : op_trait_base< ConcreteType, Impl >
        {
            static logical_result verifyRegionTrait(operation /* op */) {
                return mlir::success();
            }
        };
    };

    string_ref symbol_attr_name();

} // namespace vast::core
