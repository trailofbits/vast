// Copyright (c) 2021-present, Trail of Bits, Inc.

#pragma once

#include <vast/Util/Common.hpp>

namespace vast
{
    bool is_trailing_scope(operation op) {
        if (!mlir::isa< hl::ScopeOp >(op))
            return false;
        if (auto parent = op->getParentRegion()) {
            if(parent->hasOneBlock()) {
                auto &block = parent->back();
                // check if we have only the scope operation in the block
                return &block.front() == &block.back();
            }
        }
        return false;
    }
} //namespace vast
