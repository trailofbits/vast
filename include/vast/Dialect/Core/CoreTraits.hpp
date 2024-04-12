// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <mlir/IR/OpDefinition.h>
VAST_UNRELAX_WARNINGS

#include "vast/Util/Common.hpp"

namespace vast::core
{
    template< typename ConcreteType, template< typename > class Derived >
    using op_trait_base = mlir::OpTrait::TraitBase< ConcreteType, Derived >;

    template< typename ConcreteType, template< typename > class Derived >
    using attr_trait_base = mlir::AttributeTrait::TraitBase< ConcreteType, Derived >;

    //
    // SoftTerminatorTrait
    //
    template< typename ConcreteType >
    struct SoftTerminatorTrait : op_trait_base< ConcreteType, SoftTerminatorTrait > {};

    static inline bool is_soft_terminator(operation op) {
        return op->hasTrait< SoftTerminatorTrait >();
    }

    //
    // ReturnLikeTrait
    //
    template< typename ConcreteType >
    struct ReturnLikeTrait : op_trait_base< ConcreteType, ReturnLikeTrait > {};

    static inline bool is_return_like(operation op) {
        return op->hasTrait< ReturnLikeTrait >();
    }
} // namespace vast::core

