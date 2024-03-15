// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

#include "vast/Util/Common.hpp"

#include "vast/Dialect/Core/CoreOps.hpp"
#include "vast/Dialect/Core/CoreAttributes.hpp"

#include "vast/Dialect/HighLevel/HighLevelOps.hpp"
#include "vast/Dialect/HighLevel/HighLevelAttributes.hpp"
#include "vast/Dialect/HighLevel/HighLevelTypes.hpp"

namespace vast::cg {

    template< typename scope_t >
    struct scope_generator_guard : insertion_guard {
        scope_generator_guard(mlir_builder &builder, loc_t loc)
            : insertion_guard(builder), loc(loc)
            , scope(builder.create< scope_t >(loc))
        {
            auto &block = scope.getBody().emplaceBlock();
            builder.setInsertionPointToStart(&block);
        }

        scope_t get() { return scope; }

        loc_t loc;
        scope_t scope;
    };

    using CoreScope = scope_generator_guard< core::ScopeOp >;
    using TranslationUnitScope = scope_generator_guard< hl::TranslationUnitOp >;

    //
    // composable builder state
    //
    template< typename result_type, typename bind_type >
    struct compose_state_t;

    template< typename result_type, typename bind_type >
    struct compose_state_t {

        compose_state_t(bind_type &&binder) : binder(std::forward< bind_type >(binder)) {}

        template< typename arg_t >
        static constexpr bool valid(const arg_t &arg) {
            if constexpr (std::convertible_to< arg_t , bool >) {
                return static_cast< bool >(arg);
            } else {
                // initialized non-boolean arg is always valid
                return true;
            }
        }

        template< typename ...args_t >
        static constexpr bool valid(const args_t &...args) { return (valid(args) && ...); }

        template< typename ...args_t >
        constexpr inline auto bind(args_t &&...args) && {
            auto binded = [... args = std::forward< args_t >(args), binder = std::move(binder)] (auto &&...rest) {
                if (!valid(args...)) {
                    return result_type{};
                }
                return binder(args..., std::forward< decltype(rest) >(rest)...);
            };
            return compose_state_t< result_type, decltype(binded) >(std::move(binded));
        }

        template< typename arg_t >
        constexpr inline auto bind_if(bool cond, arg_t &&arg) && {
            auto binded = [cond, arg = std::forward< arg_t >(arg), binder = std::move(binder)] (auto &&...args) {
                if (cond) {
                    if (!valid(arg)) {
                        return result_type{};
                    }

                    return binder(arg, std::forward< decltype(args) >(args)...);
                }

                return binder(std::forward< decltype(args) >(args)...);
            };
            return compose_state_t< result_type, decltype(binded) >(std::move(binded));
        }

        template< typename arg_t >
        constexpr inline auto bind_if_valid(arg_t &&arg) && {
            auto binded = [arg = std::forward< arg_t >(arg), binder = std::move(binder)] (auto &&...args) {
                if (valid(arg)) {
                    return binder(arg, std::forward< decltype(args) >(args)...);
                }

                return binder(std::forward< decltype(args) >(args)...);
            };
            return compose_state_t< result_type, decltype(binded) >(std::move(binded));
        }

        template< typename arg_t >
        constexpr inline auto bind_region_if(bool cond, arg_t &&arg) && {
            auto binded = [cond, arg = std::forward< arg_t >(arg), binder = std::move(binder)] (auto &&...args) {
                if (cond) {
                    if (!valid(arg)) {
                        return result_type{};
                    }

                    return binder(arg, std::forward< decltype(args) >(args)...);
                }
                return binder(std::nullopt, std::forward< decltype(args) >(args)...);
            };
            return compose_state_t< result_type, decltype(binded) >(std::move(binded));
        }

        auto freeze() { return binder(); }

        bind_type binder;
    };

    struct codegen_builder : mlir_builder {
        using mlir_builder::mlir_builder;

        template< typename result_type, typename builder_type >
        auto compose_start(builder_type &&builder) {
            return compose_state_t< result_type, builder_type >(std::forward< builder_type >(builder));
        }

        template< typename op_t >
        auto compose() {
            return compose_start< op_t >([&] (auto&& ...args) {
                return create< op_t >(std::forward< decltype(args) >(args)...);
            });
        }
    };

} // namespace vast::cg
