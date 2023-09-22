// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <mlir/IR/OpDefinition.h>
VAST_UNRELAX_WARNINGS

namespace vast::core
{
    template< typename concrete_t >
    struct soft_terminator : mlir::OpTrait::TraitBase< concrete_t,
                                                       soft_terminator >
    {};

    static inline bool is_soft_terminator( mlir::Operation *op )
    {
        return op->hasTrait< soft_terminator >();
    }

    template< typename concrete_t >
    struct return_trait : mlir::OpTrait::TraitBase< concrete_t,
                                                    return_trait >
    {};

    static inline bool is_return( mlir::Operation *op )
    {
        return op->hasTrait< return_trait >();
    }
} // namespace vast::core

