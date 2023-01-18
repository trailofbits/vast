// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <mlir/Support/LogicalResult.h>
VAST_UNRELAX_WARNINGS

#include "vast/Util/Common.hpp"
namespace vast {

    template< typename From, typename Value >
    struct ScopedValueTable : llvm::ScopedHashTable< From, Value > {
        using ValueType = Value;

        using Base = llvm::ScopedHashTable< From, Value >;
        using Base::Base;

        logical_result declare(From from, Value value) {
            if (this->count(from))
                return mlir::failure();
            this->insert(from, value);
            return mlir::success();
        }
    };

} // namespace vast
