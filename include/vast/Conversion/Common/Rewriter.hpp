// Copyright (c) 2023-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include "mlir/Transforms/DialectConversion.h"
VAST_UNRELAX_WARNINGS

#include "vast/Util/Common.hpp"

#include <gap/coro/generator.hpp>

#include <ranges>
#include <vector>

namespace vast::conv
{
    template< typename impl_t >
    struct rewriter_wrapper_t
    {
        using self_t = rewriter_wrapper_t< impl_t >;
        using underlying_t = impl_t;

        // Sadly, it looks like mlir does not expose a generic way to retrieve guard per
        // builder type.
        using guard_t = InsertionGuard;

        underlying_t &bld;

        rewriter_wrapper_t( underlying_t &bld ) : bld( bld ) {}

        auto &operator*() { return bld; }
        const auto &operator*() const { return bld; }

        auto *operator->() { return &bld; }
        const auto *operator->() const { return &bld; }

        auto guarded( auto &&fn )
        {
            auto g = guard();
            return fn();
        }

        [[nodiscard]] auto guard() { return guard_t( bld ); }

        auto guarded_at_end( mlir::Block *block, auto &&fn )
        {
            auto g = guard();
            bld.setInsertionPointToEnd( block );
            return fn();
        }

        template< typename Trg, typename Bld, typename ... Args >
        auto make_after_op( Operation *op, Args && ... args )
        {
            auto g = guard();
            bld.setInsertionPointAfter( op );
            return bld.template create< Trg >( std::forward< Args >( args ) ... );
        }

        template< typename Trg, typename ... Args >
        auto make_at_end( mlir::Block *block, Args && ... args )
        {
            auto g = guard();
            bld.setInsertionPointToEnd( block );
            return bld.template create< Trg >( std::forward< Args >( args ) ... );
        }

        template< typename T >
        void safe_erase( gap::generator< T > &&range )
        {
            // It is highly probable there is an ongoing iteration
            // inside the generator, therefore we first freeze the value.
            // TODO(c++23): Refactor.
            std::vector< T > to_erase;
            std::ranges::move(range, std::back_inserter(to_erase));
            return erase( to_erase );
        }

        template< typename O >
        void erase( const std::vector< O > &to_erase )
        {
            for ( auto o : to_erase )
                bld.eraseOp( o );
        }
    };

    template< typename I >
    rewriter_wrapper_t( I & ) -> rewriter_wrapper_t< I >;

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
