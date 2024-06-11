// Copyright (c) 2024-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <llvm/ADT/Hashing.h>
#include <mlir/IR/BuiltinAttributes.h>
VAST_UNRELAX_WARNINGS

#include <variant>

namespace std {
    template<>
    struct hash< mlir::StringAttr >
    {
        size_t operator()(const mlir::StringAttr &attr) const noexcept {
            return hash_value(attr);
        }
    };

    template< typename... Ts >
    inline ::llvm::hash_code hash_value(const std::variant< Ts... > &value) {
        return ::llvm::hash_code(std::hash< std::variant< Ts... > >{}(value));
    }
} // namespace std
