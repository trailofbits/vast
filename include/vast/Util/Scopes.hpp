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
                auto last = --block.end();

                // vast-cc adds UnreachableOp if it doesn't see a proper terminator
                // But the real terminator might be enclosed in the scope
                if (mlir::isa< hl::UnreachableOp >(block.back()))
                        --last;
                // check if we have only the scope operation in the block
                return block.begin() == last;
            }
        }
        return false;
    }
} //namespace vast
