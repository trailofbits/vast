// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

#include <limits>
#include <optional>
#include <type_traits>

namespace vast
{
    // Really simple wrapper around "optional" types such as `std::optional`, or
    // `mlir::Type`.
    // Use-case is to avoid explicit branches to check values, when return value at the end
    // is some "optional" type or bool anyway.
    // TODO(lukas): Whether it is worth to use is still an open question, as a lot of stuff
    //              is methods on `mlir::` objects and that requires extra wrapping ending
    //              in verbose code again.
    //              One possible outcome is to define specific transformers with predefined ops
    //              and work/extend those.
    // TODO(lukas): For now, underlying object is always copied-moved, maybe there
    //              is some possibility for improvement.
    template< class T >
    struct Maybe
    {
        static_assert(std::is_default_constructible_v< T >);
        static_assert(std::is_move_constructible_v< T >);
        T self;

        explicit Maybe() = default;
        explicit Maybe(T self_)
            : self( std::move(self_ ) )
        {}

        template< class F >
        auto transform(F &&f)
        {
            return Maybe( f(self) );
        }

        template< class F >
        auto and_then(F &&f)
        {
            using rt = decltype( f(self) );
            return (has_value()) ? Maybe< rt >( f(self) ) : Maybe< rt >();
        }

        template< class F >
        auto or_else(F &&f)
        {
            return (has_value()) ? Maybe() : Maybe( f(self) );
        }

        template< class F = bool(*)(T) >
        auto keep_if(F &&f)
        {
            return (has_value() && f(self)) ? Maybe( std::move(*this) ) : Maybe();
        }

        template< template < class > class W >
        W< T > take_wrapped()
        {
            return (has_value()) ? W< T >( take() ) : W< T >();
        }

        template< typename W >
        W take_wrapped()
        {
            return (has_value()) ? W( take() ) : W();
        }

        auto unwrap()
        {
            // Overall getting value_type is tricky, as types tend to have very
            // different APIs.
            using rt = typename T::value_type;
            return (has_value()) ? Maybe< rt >( std::move( *self ) ) : Maybe< rt >();
        }

        auto take() { return std::move( self ); }

        explicit operator bool() const { return static_cast< bool >(self); }
        bool has_value() const
        {
            return static_cast< bool >(self);
        }
    };

    template< typename T >
    auto dyn_cast = [] (auto arg) { return mlir::dyn_cast< T >(arg); };

    template< typename T >
    auto cast = [] (auto arg) { return mlir::dyn_cast< T >(arg); };

    template< typename T >
    Maybe(T t) -> Maybe< T >;

} // namespace vast
