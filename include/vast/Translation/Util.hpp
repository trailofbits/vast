// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Region.h>
VAST_UNRELAX_WARNINGS

#include "vast/Dialect/HighLevel/HighLevelDialect.hpp"

namespace vast::cg
{
    inline void splice_trailing_scopes(mlir::Region::BlockListType &blocks) {
        auto has_trailing_scope = [&] {
            if (blocks.empty())
                return false;
            auto &last_block = blocks.back();
            if (last_block.empty())
                return false;
            return mlir::isa< hl::ScopeOp >(last_block.back());
        };

        while (has_trailing_scope()) {
            auto &last_block = blocks.back();

            auto scope  = mlir::cast< hl::ScopeOp >(last_block.back());
            auto parent = scope.getBody().getParentRegion();
            scope->remove();

            auto &prev = parent->getBlocks().back();

            mlir::BlockAndValueMapping mapping;
            scope.getBody().cloneInto(parent, mapping);

            auto next = prev.getNextNode();

            auto &ops = last_block.getOperations();
            ops.splice(ops.end(), next->getOperations());

            next->erase();
            scope.erase();
        }
    }

    inline void splice_trailing_scopes(hl::FuncOp &fn) {
        if (fn.empty())
            return;
        splice_trailing_scopes(fn.getBlocks());
    }

    inline void splice_trailing_scopes(mlir::Region &reg) {
        splice_trailing_scopes(reg.getBlocks());
    }

} // namespace vast::cg
