// Copyright (c) 2021-present, Trail of Bits, Inc.

#pragma once

#include <vast/Util/Common.hpp>

namespace vast
{
    bool has_trailing_scope(mlir::Region::BlockListType &blocks) {
        if (blocks.empty())
            return false;
        auto &last_block = blocks.back();
        if (last_block.empty())
            return false;
        return mlir::isa< hl::ScopeOp >(last_block.back());
    }

    bool has_trailing_scope(Region &r) { return has_trailing_scope(r.getBlocks()); }

    bool has_trailing_scope(operation op){
        for (auto &r : op->getRegions())
            if (has_trailing_scope(r))
                return true;
        return false;
    }
} //namespace vast
