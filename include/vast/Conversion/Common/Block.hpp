// Copyright (c) 2023-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include "mlir/Transforms/DialectConversion.h"
VAST_UNRELAX_WARNINGS

namespace vast::conv
{
    template< typename Op, typename Builder >
    auto extract_as_block( Op op, Builder &bld )
        -> std::tuple< mlir::Block *, mlir::Block *, mlir::Block * >
    {
        auto it = mlir::Block::iterator( op );
        auto head = op->getBlock();

        auto body = bld.splitBlock( head, it );
        ++it;

        auto tail = bld.splitBlock( body, it );
        VAST_CHECK( head && body && tail, "Extract instruction as solo block failed." );
        return { head, body, tail };
    }

    template< typename Op, typename Builder >
    auto split_at_op( Op op, Builder &bld )
    {
        auto it = mlir::Block::iterator( op );
        auto head = op->getBlock();

        auto body = bld.splitBlock( head, it );

        return std::make_tuple( head, body );
    }

    template< typename Op, typename Bld >
    mlir::Block *inline_region( Op op, Bld &bld, mlir::Region &region, mlir::Region &dest )
    {
        auto begin = &region.front();
        auto end   = &region.back();
        VAST_CHECK( begin == end, "Region has more than one block" );

        bld.inlineRegionBefore( region, dest, dest.end() );
        return begin;
    }

    template< typename bld_t >
    void inline_region_blocks(bld_t &bld,
                              mlir::Region &region,
                              mlir::Region::iterator before)
    {
        bld.inlineRegionBefore(region, *before->getParent(), before);
    }

} // namespace vast::conv
