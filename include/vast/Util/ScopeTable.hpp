// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <mlir/Support/LogicalResult.h>
VAST_UNRELAX_WARNINGS

namespace vast {

    using LogicalResult = mlir::LogicalResult;
    using StringRef     = llvm::StringRef; 
    
    template< typename From, typename Value >
    struct ScopedValueTable : llvm::ScopedHashTable< From, Value > {
        using ValueType = Value;

        using Base = llvm::ScopedHashTable< From, Value >;
        using Base::Base;

        LogicalResult declare(From from, Value value) {
            if (this->count(from))
                return mlir::failure();
            this->insert(from, value);
            return mlir::success();
        }
    };


    template< typename Value >
    struct ScopedSymbolTable : ScopedValueTable< StringRef, Value > 
    {
        using Base = ScopedValueTable< StringRef, Value >;
        using Base::Base;
    };

} // namespace vast
