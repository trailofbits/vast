// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Region.h>
VAST_UNRELAX_WARNINGS

#include <gap/core/generator.hpp>

namespace vast::cg
{
    template< typename T >
    gap::generator< T * > filter(auto from) {
        for (auto x : from) {
            if (auto s = clang::dyn_cast< T >(x))
                co_yield s;
        }
    }
} // namespace vast::cg
