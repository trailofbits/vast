// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

#include "vast/Util/Common.hpp"
#include "vast/Util/Maybe.hpp"

#include "vast/Dialect/Core/CoreOps.hpp"
#include "vast/Dialect/Core/CoreAttributes.hpp"

#include "vast/Dialect/HighLevel/HighLevelOps.hpp"
#include "vast/Dialect/HighLevel/HighLevelAttributes.hpp"
#include "vast/Dialect/HighLevel/HighLevelTypes.hpp"

namespace vast::cg {

    static inline auto first_result = [] (auto op) { return op->getResult(0); };
    static inline auto results = [] (auto op) { return op->getResults(); };

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

        template< typename arg_t >
        constexpr inline auto bind(std::optional< arg_t > &&arg) && {
            auto bound = [arg = std::move(arg), binder = std::move(binder)] (auto &&...rest) {
                if (!arg.has_value() || !valid(arg.value())) {
                    return result_type{};
                }
                return binder(std::move(arg.value()), std::forward< decltype(rest) >(rest)...);
            };
            return compose_state_t< result_type, decltype(bound) >(std::move(bound));
        }

        template< typename ...args_t >
        constexpr inline auto bind_always(args_t &&...args) && {
            auto bound = [... args = std::forward< args_t >(args), binder = std::move(binder)] (auto &&...rest) {
                return binder(args..., std::forward< decltype(rest) >(rest)...);
            };
            return compose_state_t< result_type, decltype(bound) >(std::move(bound));
        }


        template< typename ...args_t >
        constexpr inline auto bind(args_t &&...args) && {
            auto bound = [... args = std::forward< args_t >(args), binder = std::move(binder)] (auto &&...rest) {
                if (!valid(args...)) {
                    return result_type{};
                }
                return binder(args..., std::forward< decltype(rest) >(rest)...);
            };
            return compose_state_t< result_type, decltype(bound) >(std::move(bound));
        }

        template< typename target_t, typename arg_t >
        constexpr inline auto bind_dyn_cast(arg_t &&arg) && {
            auto bound = [arg = std::forward< arg_t >(arg), binder = std::move(binder)] (auto &&...rest) {
                if (!valid(arg)) {
                    return result_type{};
                }

                if (auto casted = dyn_cast< target_t >(arg)) {
                    return binder(casted, std::forward< decltype(rest) >(rest)...);
                }

                return result_type{};
            };
            return compose_state_t< result_type, decltype(bound) >(std::move(bound));
        }

        template< typename arg_t >
        constexpr inline auto bind_if(bool cond, arg_t &&arg) && {
            auto bound = [cond, arg = std::forward< arg_t >(arg), binder = std::move(binder)] (auto &&...args) {
                if (cond) {
                    if (!valid(arg)) {
                        return result_type{};
                    }

                    return binder(arg, std::forward< decltype(args) >(args)...);
                }

                return binder(std::forward< decltype(args) >(args)...);
            };
            return compose_state_t< result_type, decltype(bound) >(std::move(bound));
        }

        template< typename arg_t >
        constexpr inline auto bind_if_valid(arg_t &&arg) && {
            auto bound = [arg = std::forward< arg_t >(arg), binder = std::move(binder)] (auto &&...args) {
                if (valid(arg)) {
                    return binder(arg, std::forward< decltype(args) >(args)...);
                }

                return binder(std::forward< decltype(args) >(args)...);
            };
            return compose_state_t< result_type, decltype(bound) >(std::move(bound));
        }

        template< typename then_arg_t, typename else_arg_t >
        constexpr inline auto bind_choose(bool cond, then_arg_t &&then_arg, else_arg_t &&else_arg) && {
            auto bound = [cond, then_arg = std::forward<then_arg_t>(then_arg), else_arg = std::forward<else_arg_t>(else_arg), binder = std::move(binder)](auto &&...args) {
                if (cond) {
                    if (!valid(then_arg)) {
                        return result_type{};
                    }
                    return binder(then_arg, std::forward<decltype(args)>(args)...);
                } else {
                    if (!valid(else_arg)) {
                        return result_type{};
                    }
                    return binder(else_arg, std::forward<decltype(args)>(args)...);
                }
            };

            return compose_state_t<result_type, decltype(bound)>(std::move(bound));
        }

        template< typename arg_t, typename func_t >
        constexpr inline auto bind_transform(arg_t &&arg, func_t &&fn) && {
            auto bound = [
                arg = std::forward< arg_t >(arg),
                fn = std::forward< func_t >(fn),
                binder = std::move(binder)
            ] (auto &&...args) {
                if (!valid(arg)) {
                    return result_type{};
                }

                return binder(fn(arg), std::forward< decltype(args) >(args)...);
            };
            return compose_state_t< result_type, decltype(bound) >(std::move(bound));
        }

        auto freeze() { return binder(); }

        auto freeze_as_maybe() {
            return Maybe< result_type >(binder());
        }

        bind_type binder;
    };

    struct codegen_builder : mlir_builder {
        using mlir_builder::mlir_builder;

        insertion_guard insertion_guard() { return { *this }; }

        void set_insertion_point_to_start(region_ptr region) {
            set_insertion_point_to_start(&region->front());
        }

        void set_insertion_point_to_end(region_ptr region) {
            set_insertion_point_to_end(&region->back());
        }

        void set_insertion_point_to_start(block_ptr block) {
            setInsertionPointToStart(block);
        }

        void set_insertion_point_to_end(block_ptr block) {
            setInsertionPointToEnd(block);
        }

        [[nodiscard]] auto scoped_insertion_at_start(block_ptr block) {
            auto guard = insertion_guard();
            set_insertion_point_to_start(block);
            return guard;
        }

        [[nodiscard]] auto scoped_insertion_at_end(block_ptr block) {
            auto guard = insertion_guard();
            set_insertion_point_to_end(block);
            return guard;
        }

        [[nodiscard]] auto scoped_insertion_at_start(region_ptr region) {
            VAST_CHECK(!region->empty(), "Inserting into region with no blocks");
            return scoped_insertion_at_start(&region->front());
        }

        [[nodiscard]] auto scoped_insertion_at_end(region_ptr region) {
            VAST_CHECK(!region->empty(), "Inserting into region with no blocks");
            return scoped_insertion_at_end(&region->back());
        }

        [[nodiscard]] auto get_module_region() {
            auto curr = getBlock()->getParentOp();
            while (!mlir::isa< vast_module >(curr)) {
                curr = curr->getParentOp();
            }
            return mlir::cast< vast_module >(curr).getBody();
        }

        [[nodiscard]] auto set_insertion_point_to_start_of_module() {
            return scoped_insertion_at_start(get_module_region());
        }

        [[nodiscard]] auto set_insertion_point_to_end_of_module() {
            return scoped_insertion_at_end(get_module_region());
        }

        hl::VoidType void_type() { return getType< hl::VoidType >(); }
        hl::BoolType bool_type() { return getType< hl::BoolType >(); }

        mlir_value void_value(loc_t loc) {
            return create< hl::ConstantOp >(loc, void_type());
        }

        mlir_value bool_value(loc_t loc, bool value) {
            return create< hl::ConstantOp >(loc, bool_type(), value);
        }

        mlir_value true_value(loc_t loc)  { return bool_value(loc, true); }
        mlir_value false_value(loc_t loc) { return bool_value(loc, false); }

        mlir_value constant(loc_t loc) {
            return void_value(loc);
        }

        mlir_value constant(loc_t loc, bool value) {
            return bool_value(loc, value);
        }

        mlir_value constant(loc_t loc, mlir_type ty, auto &&value) {
            return create< hl::ConstantOp >(loc, ty, std::forward< decltype(value) >(value));
        }

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

        template< typename T >
        Type bitwidth_type() {
            return mlir::IntegerType::get(getContext(), bits< T >());
        }

        template< typename T >
        integer_attr_t interger_attr(T v) {
            return integer_attr_t::get(bitwidth_type< T >(), v);
        }

        integer_attr_t  u8(uint8_t  v) { return interger_attr(v); }
        integer_attr_t u16(uint16_t v) { return interger_attr(v); }
        integer_attr_t u32(uint32_t v) { return interger_attr(v); }
        integer_attr_t u64(uint64_t v) { return interger_attr(v); }

        integer_attr_t  i8(int8_t  v) { return interger_attr(v); }
        integer_attr_t i16(int16_t v) { return interger_attr(v); }
        integer_attr_t i32(int32_t v) { return interger_attr(v); }
        integer_attr_t i64(int64_t v) { return interger_attr(v); }
    };


} // namespace vast::cg
