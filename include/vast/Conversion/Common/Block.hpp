// Copyright (c) 2023-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include "mlir/Transforms/DialectConversion.h"
VAST_UNRELAX_WARNINGS

namespace vast::conv {
    // Give operation `op` in the block:
    // ```
    // head = ...
    // op = ...
    // tail = ...
    // ```
    // Each of `head, op, tail` are in separate blocks ( no control flow transfer is injected )
    // and return value are pointers to these newly formed blocks.
    // `[ head_block, op_block, tail_block ]`.
    template< typename op_t, typename bld_t >
    auto extract_as_block(op_t op, bld_t &bld)
        -> std::tuple< mlir::Block *, mlir::Block *, mlir::Block * > {
        auto it   = mlir::Block::iterator(op);
        auto head = op->getBlock();

        auto body = bld.splitBlock(head, it);
        ++it;

        auto tail = bld.splitBlock(body, it);
        VAST_CHECK(head && body && tail, "Extract instruction as solo block failed.");
        return { head, body, tail };
    }

    // Give operation `op` in the block:
    // ```
    // head = ...
    // op = ...
    // tail = ...
    // ```
    // Split the block into two:
    // ```
    // bb1:
    //   head
    // bb2:
    //   op
    //   tail
    // ```
    // Returned value are pointers `[ bb1, bb2 ]`.
    template< typename op_t, typename bld_t >
    auto split_at_op(op_t op, bld_t &bld) {
        auto it   = mlir::Block::iterator(op);
        auto head = op->getBlock();

        auto body = bld.splitBlock(head, it);

        return std::make_tuple(head, body);
    }

    template< typename op_t, typename bld_t >
    auto split_after_op(op_t op, bld_t &bld) {
        auto it   = mlir::Block::iterator(op);
        auto head = op->getBlock();

        auto body = bld.splitBlock(head, ++it);

        return std::make_tuple(head, body);
    }

    template< typename bld_t >
    mlir::Block *inline_region(bld_t &bld, mlir::Region &region, mlir::Region &dest) {
        auto begin = &region.front();
        auto end   = &region.back();
        VAST_CHECK(begin == end, "Region has more than one block");

        bld.inlineRegionBefore(region, dest, dest.end());
        return begin;
    }

    template< typename bld_t >
    void inline_region_blocks(bld_t &bld, mlir::Region &region, mlir::Region::iterator before) {
        bld.inlineRegionBefore(region, *before->getParent(), before);
    }

    static inline std::size_t size(mlir::Block &block) {
        return static_cast< size_t >(std::distance(block.begin(), block.end()));
    }

    static inline std::size_t size(mlir::Region &region) {
        return static_cast< size_t >(std::distance(region.begin(), region.end()));
    }

    static inline bool empty(mlir::Block &block) { return size(block) == 0; }

    static inline bool empty(mlir::Region &region) { return size(region) == 0; }

} // namespace vast::conv
