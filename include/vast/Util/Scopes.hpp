// Copyright (c) 2021-present, Trail of Bits, Inc.

#pragma once

#include <vast/Util/Common.hpp>

namespace vast
{
    inline bool is_trailing_scope(operation op) {
        if (!mlir::isa< hl::ScopeOp >(op))
            return false;
        if (auto parent = op->getParentRegion()) {
            if(parent->hasOneBlock()) {
                auto &block = parent->back();
                auto last = --block.end();
                return block.begin() == last;
            }
        }
        return false;
    }

    inline operation get_last_op(hl::ScopeOp scope) {
        auto &last_block = scope.getBody().back();
        if (last_block.empty())
            return nullptr;
        return &last_block.back();
    }
} //namespace vast
