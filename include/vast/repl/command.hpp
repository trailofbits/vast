// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <llvm/ADT/StringExtras.h>
VAST_UNRELAX_WARNINGS

#include "vast/repl/state.hpp"
#include "vast/repl/common.hpp"

#include "vast/Util/Tuple.hpp"
#include "vast/Util/TypeList.hpp"
#include "vast/repl/codegen.hpp"

#include <filesystem>
#include <tuple>
#include <span>

namespace vast::repl
{
    using logical_result = mlir::LogicalResult;

    using string_ref     = llvm::StringRef;
    using command_token  = string_ref;
    using command_tokens = llvm::SmallVector< command_token >;

    namespace command
    {
        struct base {
            using command_params = util::type_list<>;

            virtual void run(state_t &) const = 0;
            virtual ~base(){};
        };

        struct file_param { std::filesystem::path path; };
        struct flag_param { bool set; };

        enum class show_kind { source, ast, module };

        template< typename enum_type >
        enum_type from_string(string_ref token) requires(std::is_same_v< enum_type, show_kind >) {
            if (token == "source")  return enum_type::source;
            if (token == "ast")     return enum_type::ast;
            if (token == "module")  return enum_type::module;
            VAST_UNREACHABLE("uknnown show kind: {}", token.str());
        }

        //
        // named param
        //
        template< const char *name, typename base >
        struct named_param {
            static constexpr inline const char *param_name = name;

            static constexpr bool is_file_param = std::is_same_v< base, file_param >;
            static named_param parse(string_ref token) requires(is_file_param) {
                return { .value = file_param{ token.str() } };
            }

            static constexpr bool is_flag_param = std::is_same_v< base, flag_param >;
            static named_param parse(string_ref token) requires(is_flag_param) {
                return { .value = flag_param(token == param_name) };
            }

            static constexpr bool is_enum_param = std::is_enum_v< base >;
            static named_param parse(string_ref token) requires(is_enum_param) {
                return { .value = from_string< base >(token) };
            }

            base value;
        };

        template< const char *name, typename params_storage >
        const auto &get_param(const params_storage &params) {
            if constexpr (std::tuple_size_v< params_storage > == 0) {
                throw std::runtime_error(("unknown param name " + std::string(name)));
            } else {
                using current = typename std::tuple_element< 0, params_storage >::type;

                if constexpr (current::param_name == name) {
                    return util::head(params).value;
                } else {
                    return get_param< name >(util::tail(params));
                }
            }
        }

        //
        // exit command
        //
        struct exit : base {
            static constexpr string_ref name() { return "exit"; }

            using base::command_params;

            void run(state_t &state) const override {
                state.exit = true;
            };
        };


        //
        // help command
        //
        struct help : base {
            static constexpr string_ref name() { return "help"; }

            using base::command_params;

            void run(state_t&) const override {
                throw std::runtime_error("help not implemented");
            };
        };

        //
        // load command
        //
        struct load : base {
            static constexpr string_ref name() { return "load"; }

            static constexpr inline char source_param_name[] = "source";

            using command_params = util::type_list<
                named_param< source_param_name, file_param >
            >;

            using params_storage = command_params::as_tuple;

            load(const params_storage &params) : params(params) {}
            load(params_storage &&params) : params(std::move(params)) {}

            void run(state_t &state) const override {
                auto source  = get_param< source_param_name >(params);
                state.source = codegen::get_source(source.path);
            };

            params_storage params;
        };

        //
        // show command
        //
        struct show : base {
            static constexpr string_ref name() { return "show"; }

            static constexpr inline char kind_param_name[] = "kind_param_name";

            using command_params = util::type_list<
                named_param< kind_param_name, show_kind >
            >;

            using params_storage = command_params::as_tuple;

            show(const params_storage &params) : params(params) {}
            show(params_storage &&params) : params(std::move(params)) {}

            void run(state_t &state) const override {
                auto what = get_param< kind_param_name >(params);
                switch (what) {
                    case show_kind::source: return show_source(state);
                    case show_kind::ast:    return show_ast(state);
                    case show_kind::module: return show_module(state);
                }
            };

            void check_source(const state_t &state) const {
                if (!state.source.has_value()) {
                    throw std::runtime_error("error: missing source");
                }
            }

            const std::string &get_source(const state_t &state) const {
                check_source(state);
                return state.source.value();
            }

            void show_source(const state_t &state) const {
                llvm::outs() << get_source(state);
            }

            void show_ast(const state_t &state) const {
                auto unit = codegen::ast_from_source(get_source(state));
                unit->getASTContext().getTranslationUnitDecl()->dump(llvm::outs());
            }

            void show_module(state_t &state) const {
                if (!state.mod) {
                    const auto &source = get_source(state);
                    state.mod = codegen::emit_module(source, &state.ctx);
                }

                llvm::outs() << state.mod.get();
            }

            params_storage params;
        };

        using command_list = util::type_list< exit, help, load, show >;

    } // namespace command

    using command_ptr = std::unique_ptr< command::base >;

    static inline std::span< string_ref > tail(std::span<string_ref > tokens) {
        if (tokens.size() == 1)
            return {};
        return tokens.subspan(1);
    }

    static inline command_tokens parse_tokens(string_ref cmd) {
        command_tokens tokens;
        llvm::SplitString(cmd, tokens);
        return tokens;
    }

    template< typename params_list >
    auto parse_params(std::span< string_ref > tokens)
        -> typename params_list::as_tuple
    {
        if (tokens.empty()) {
            return {};
        }

        if constexpr (params_list::empty) {
            throw std::runtime_error(("no match for param: " + tokens.front()).str());
        } else {
            using current_param = typename params_list::head;
            using rest          = typename params_list::tail;

            auto param = std::make_tuple(current_param::parse(tokens.front()));
            return std::tuple_cat(param, parse_params< rest >(tail(tokens)));
        }
    }

    template< typename command, typename params_storage >
    command_ptr make_command(params_storage &&params) {
        constexpr auto params_size = std::tuple_size_v< std::remove_reference_t< params_storage > >;
        if constexpr (params_size != 0) {
            return std::make_unique< command >(std::forward< params_storage >(params));
        } else {
            return std::make_unique< command >();
        }
    }

    template< typename commands >
    command_ptr match(std::span< string_ref > tokens) {
        if (tokens.empty()) {
            // print help, if no command was provided
            return std::make_unique< command::help >();
        }

        if constexpr (commands::empty) {
            // we did not recursivelly match any of known commands
            throw std::runtime_error(("no match for command: " + tokens.front()).str());
        } else {
            using current_command = typename commands::head;
            using rest            = typename commands::tail;
            using command_params  = typename current_command::command_params;

            // match current command
            if (current_command::name() == tokens.front()) {
                auto params = parse_params< command_params >(tail(tokens));
                return make_command< current_command >(std::move(params));
            }

            // match other command from the list of available commands
            return match< rest >(tokens);
        }
    }

    static inline command_ptr parse_command(std::span< command_token > tokens) {
        return match< command::command_list >(tokens);
    }

} // namespace vast::repl
