// Copyright (c) 2023-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include "mlir/Transforms/DialectConversion.h"
VAST_UNRELAX_WARNINGS

namespace vast::conv
{
    auto guarded( auto &bld, auto &&fn )
    {
        auto g = mlir::OpBuilder::InsertionGuard( bld );
        return fn();
    }

    auto guarded_at_end( auto &bld, mlir::Block *block, auto &&fn )
    {
        auto g = mlir::OpBuilder::InsertionGuard( bld );
        bld.setInsertionPointToEnd( block );
        return fn();
    }

    template< typename Trg, typename Bld, typename ... Args >
    auto make_after_op( Bld &bld, Operation *op, Args && ... args )
    {
        mlir::OpBuilder::InsertionGuard guard( bld );
        bld.setInsertionPointAfter( op );
        return bld.template create< Trg >( std::forward< Args >( args ) ... );
    }

    template< typename Trg, typename Bld, typename ... Args >
    auto make_at_end( Bld &bld, mlir::Block *block, Args && ... args )
    {
        mlir::OpBuilder::InsertionGuard guard( bld );
        bld.setInsertionPointToEnd( block );
        return bld.template create< Trg >( std::forward< Args >( args ) ... );
    }
} // namespace vast::conv
