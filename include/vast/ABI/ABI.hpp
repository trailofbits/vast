// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Value.h>

#include <llvm/Support/raw_ostream.h>
VAST_UNRELAX_WARNINGS

#include "vast/Dialect/HighLevel/HighLevelTypes.hpp"
#include "vast/Util/Common.hpp"

#include <string>
#include <vector>

#include <variant>

namespace vast::abi {

    // Whole ABI transformation is heavily based on clang implementation, because there is a
    // need to be faithful to it (since once use-case is to link with clanf compiled
    // bitcode).

    // TODO(abi): Should be probably parametrized by type.
    struct passing_style_base
    {
        using type  = mlir::Type;
        using types = std::vector< type >;
    };

    struct with_target_type : passing_style_base
    {
        types target_types;

        with_target_type(type t) : target_types{ t } {}

        with_target_type(types target_types) : target_types(std::move(target_types)) {}

        with_target_type(const with_target_type &) = default;
        with_target_type(with_target_type &&)      = default;

        with_target_type &operator=(const with_target_type &) = default;
        with_target_type &operator=(with_target_type &&)      = default;

        std::string to_string() const {
            std::string out = "\ttarget_type: ";
            llvm::raw_string_ostream ss(out);
            ss << "[ ";
            for (auto t : target_types) {
                ss << t << " ";
            }
            ss << "]";
            return out;
        }
    };

    // TODO(codestyle): The formatting can be pulled into some base class.
    struct direct : with_target_type
    {
        static inline const constexpr std::string_view str = "direct";
        using with_target_type::with_target_type;

        std::string to_string() const {
            return std::string(str) + with_target_type::to_string();
        }
    };

    struct extend : with_target_type
    {
        static inline const constexpr std::string_view str = "extend";
        using with_target_type::with_target_type;

        std::string to_string() const {
            return std::string(str) + with_target_type::to_string();
        }
    };

    struct indirect : with_target_type
    {
        static inline const constexpr std::string_view str = "indirect";
        using with_target_type::with_target_type;

        std::string to_string() const {
            return std::string(str) + with_target_type::to_string();
        }
    };

    struct ignore : with_target_type
    {
        static inline const constexpr std::string_view str = "ignore";
        using with_target_type::with_target_type;

        std::string to_string() const {
            return std::string(str) + with_target_type::to_string();
        }
    };

    struct expand : with_target_type
    {
        static inline const constexpr std::string_view str = "expand";
        using with_target_type::with_target_type;

        std::string to_string() const {
            return std::string(str) + with_target_type::to_string();
        }
    };

    struct coerce_and_expand : with_target_type
    {
        static inline const constexpr std::string_view str = "coerce_and_expand";
        using with_target_type::with_target_type;

        std::string to_string() const {
            return std::string(str) + with_target_type::to_string();
        }
    };

    struct in_alloca : with_target_type
    {
        static inline const constexpr std::string_view str = "in_alloca";
        using with_target_type::with_target_type;

        std::string to_string() const {
            return std::string(str) + with_target_type::to_string();
        }
    };

    template< typename T >
    concept has_target_types = requires(const T &a) {
        {
            a.target_types
        };
    };

    using passing_style_t = std::variant<
        direct, extend, indirect, ignore, expand, coerce_and_expand,
        in_alloca
        // TODO(codestyle): Remove later, injected to make development easier.
        ,
        std::monostate >;

    struct arg_info
    {
        using type  = typename passing_style_base::type;
        using types = typename passing_style_base::types;

        passing_style_t style;

        arg_info(passing_style_t style) : style(std::move(style)) {}

        arg_info() : style(std::monostate{}) {}

        arg_info(const arg_info &) = default;
        arg_info(arg_info &&)      = default;

        arg_info &operator=(const arg_info &) = default;
        arg_info &operator=(arg_info &&)      = default;

        template< typename U, typename... Args >
        static auto make(Args &&...args) {
            auto val = passing_style_t(U(std::forward< Args >(args)...));
            return arg_info(std::move(val));
        }

        std::string to_string() const {
            return std::visit(
                [&]< typename T >(const T &e) {
                    if constexpr (std::is_same_v< T, std::monostate >) {
                        return std::string("monostate");
                    } else {
                        return e.to_string();
                    }
                },
                style
            );
        }

        types target_types() const {
            auto process = [&]< typename T >(const T &e) -> types {
                if constexpr (has_target_types< T >) {
                    return e.target_types;
                } else if (std::is_same_v< T, std::monostate >) {
                    VAST_TODO("Trying to retrieve target types from monostate");
                } else {
                    return {};
                }
            };

            return std::visit(process, style);
        }
    };

    template< typename RawFn >
    struct func_info
    {
        using args_info_t = std::vector< arg_info >;

        args_info_t _rets;
        args_info_t _args;

        // calling convention.
        RawFn raw_fn;

        using type  = typename arg_info::type;
        using types = typename arg_info::types;

        func_info(RawFn raw_fn) : raw_fn(raw_fn) {}

        std::string to_string() const {
            std::stringstream ss;
            auto fmt = [&](std::string root, const auto &collection) {
                ss << root << ":\n";
                for (const auto &info : collection) {
                    ss << "\t" << info.to_string() << "\n";
                }
            };
            fmt("return", _rets);
            fmt("args", _args);
            return ss.str();
        }

        void add_return(arg_info i) { _rets.emplace_back(std::move(i)); }

        void add_arg(arg_info i) { _args.emplace_back(std::move(i)); }

        auto fn_type() -> core::FunctionType {
            auto t = mlir::dyn_cast< core::FunctionType >(raw_fn.getFunctionType());
            VAST_CHECK(t, "{0}", raw_fn.getFunctionType());
            return t;
        }

        auto return_type() {
            auto results = fn_type().getResults();
            VAST_CHECK(results.size() == 1, "Cannot handle more return types.");
            return results[0];
        }

        const auto &args() const { return _args; }

        const auto &rets() const { return _rets; }
    };

    template< typename Op >
    func_info(Op) -> func_info< Op >;

    template< typename Fn, typename TypeInfo, typename Classifier >
    func_info< Fn > make(Fn fn, const TypeInfo &type_info) {
        auto info = func_info(fn);
        return Classifier(info, type_info).compute_abi(fn).take();
        return info;
    }

} // namespace vast::abi
