// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Region.h>
VAST_UNRELAX_WARNINGS

#include "vast/Dialect/HighLevel/HighLevelDialect.hpp"

namespace vast::cg
{
    template< typename T >
    void filter(const auto &decls, auto &&yield) {
        for ( auto decl : decls) {
            if (auto s = clang::dyn_cast< T >(decl)) {
                yield(s);
            }
        }
    }
} // namespace vast::cg
