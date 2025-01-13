// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <mlir/IR/Block.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/Operation.h>
VAST_UNRELAX_WARNINGS

#include <limits>
#include <optional>
#include <type_traits>

namespace vast {
    namespace detail {

        template< typename self_t >
        struct terminator_base
        {
            auto &self() { return static_cast< self_t & >(*this); }

            auto &self() const { return static_cast< const self_t & >(*this); }

            template< typename T >
            T cast() const {
                if (!self().has_value()) {
                    return {};
                }
                return mlir::dyn_cast< T >(self().op_ptr());
            }

            template< typename T >
            T op() const {
                auto out = self().template cast< T >();
                VAST_ASSERT(out);
                return out;
            }

            template< typename... Args >
            bool is_one_of() {
                return self().has_value() && (mlir::isa< Args >(self().op_ptr()) || ...);
            }

            static bool has(block_t &block) {
                if (std::distance(block.begin(), block.end()) == 0) {
                    return false;
                }
                return self_t::is(&block.back());
            }

            static self_t get(block_t &block) {
                if (!has(block)) {
                    return self_t{};
                }
                return self_t{ &block.back() };
            }
        };
    } // namespace detail

    static inline bool is_terminator(operation op) {
        return op->hasTrait< mlir::OpTrait::IsTerminator >();
    }

    struct hard_terminator
        : std::optional< operation >
        , detail::terminator_base< hard_terminator >
    {
        operation op_ptr() const { return **this; }
        static bool is(operation op) { return is_terminator(op); }
    };

    struct any_terminator
        : std::optional< operation >
        , detail::terminator_base< any_terminator >
    {
        operation op_ptr() const { return **this; }
        static bool is(operation op) {
            return is_terminator(op) || core::is_soft_terminator(op);
        }
    };

    template< typename op_t >
    struct terminator
        : std::optional< operation >
        , detail::terminator_base< terminator< op_t > >
    {
        using impl = detail::terminator_base< terminator< op_t > >;

        operation op_ptr() const { return **this; }

        static bool is(operation op) { return mlir::isa< op_t >(op); }

        op_t op() { return this->impl::template op< op_t >(); }
    };

} // namespace vast
