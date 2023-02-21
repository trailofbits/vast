// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <mlir/IR/Block.h>
#include <mlir/IR/Operation.h>
VAST_UNRELAX_WARNINGS

#include <limits>
#include <optional>
#include <type_traits>

namespace vast
{
    struct optional_terminator_t : std::optional< mlir::Operation * >
    {
        template< typename T >
        T cast() const
        {
            if (!has_value())
                return {};
            return mlir::dyn_cast< T >(**this);
        }

        template< typename ... Args >
        bool is_one_of() const
        {
            return has_value() && (mlir::isa< Args >( **this ) || ... );
        }
    };

    static inline bool has_terminator(mlir::Block &block)
    {
        if (std::distance(block.begin(), block.end()) == 0)
            return false;

        auto &last = block.back();
        return last.hasTrait< mlir::OpTrait::IsTerminator >();
    }

    static inline optional_terminator_t get_terminator(mlir::Block &block)
    {
        if (has_terminator(block))
            return { block.getTerminator() };
        return {};
    }
} // namespace vast
