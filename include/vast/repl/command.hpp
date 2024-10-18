// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <llvm/ADT/StringExtras.h>
VAST_UNRELAX_WARNINGS

#include "vast/Util/Tuple.hpp"
#include "vast/Util/TypeList.hpp"

#include "vast/Dialect/Meta/MetaDialect.hpp"

#include "vast/repl/state.hpp"
#include "vast/repl/common.hpp"
#include "vast/repl/codegen.hpp"

#include <filesystem>
#include <ranges>
#include <span>
#include <tuple>

namespace vast::repl
{
    using command_token  = string_ref;
    using command_tokens = llvm::SmallVector< command_token >;

    namespace cmd
    {
        void check_source(const state_t &state);

        using maybe_memory_buffer = llvm::ErrorOr< std::unique_ptr< llvm::MemoryBuffer > >;

        maybe_memory_buffer get_source_buffer(const state_t &state);

        //
        // params
        //
        struct file_param    { std::filesystem::path path; };
        struct flag_param    { bool set; };
        struct string_param  { std::string value; };
        struct integer_param { std::uint64_t value; };

        enum class show_kind { source, ast, module, symbols, pipelines, link };

        template< typename enum_type >
        enum_type from_string(string_ref token) requires(std::is_same_v< enum_type, show_kind >) {
            if (token == "source")    return enum_type::source;
            if (token == "ast")       return enum_type::ast;
            if (token == "module")    return enum_type::module;
            if (token == "symbols")   return enum_type::symbols;
            if (token == "pipelines") return enum_type::pipelines;
            if (token == "link")   return enum_type::link;
            throw_error("uknnown show kind: {0}", token.str());
        }

        enum class meta_action { add, get };

        template< typename enum_type >
        enum_type from_string(string_ref token) requires(std::is_same_v< enum_type, meta_action >) {
            if (token == "add") return enum_type::add;
            if (token == "get") return enum_type::get;
            throw_error("uknnown action kind: {0}", token.str());
        }

        void analyze_reachable_code(state_t &);
        void analyze_uninitialized_variables(state_t &);

        using analyze_func = llvm::function_ref< void(state_t &) >;

        template< typename func_type >
        func_type from_string(string_ref token) requires(std::is_same_v< analyze_func, func_type >) {
            if (token == "reachable") return analyze_reachable_code;
            if (token == "uninit")    return analyze_uninitialized_variables;
            throw_error("unknown analyze kind: {0}", token.str());
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

            static constexpr bool is_string_param = std::is_same_v< base, string_param >;
            static named_param parse(string_ref token) requires(is_string_param) {
                return { .value = { token.str() } };
            }

            static constexpr bool is_integer_param = std::is_same_v< base, integer_param >;
            static named_param parse(string_ref token) requires(is_integer_param) {
                integer_param param;
                token.getAsInteger(0, param.value);
                return { param };
            }

            static constexpr bool is_flag_param = std::is_same_v< base, flag_param >;
            static named_param parse(string_ref token) requires(is_flag_param) {
                return { .value = flag_param(token == param_name) };
            }

            static constexpr bool is_enum_param = std::is_enum_v< base >;
            static constexpr bool is_func_param = std::is_same_v< base, analyze_func >;
            static named_param parse(string_ref token) requires(is_enum_param || is_func_param) {
                return { .value = from_string< base >(token) };
            }

            base value;
        };

        template< const char *name, typename params_storage >
        auto get_param(const params_storage &params) {
            if constexpr (std::tuple_size_v< params_storage > == 0) {
                throw_error(("unknown param name " + std::string(name)).c_str());
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

            void run(state_t &state) const override;
        };

        //
        // help command
        //
        struct help : base {
            static constexpr string_ref name() { return "help"; }

            using base::command_params;

            void run(state_t&) const override;
        };

        //
        // load command
        //
        struct load : base {
            static constexpr string_ref name() { return "load"; }

            static constexpr inline char source_param[] = "source";

            using command_params = util::type_list<
                named_param< source_param, file_param >
            >;

            using params_storage = command_params::as_tuple;

            load(const params_storage &params) : params(params) {}
            load(params_storage &&params) : params(std::move(params)) {}

            void run(state_t &state) const override;

            params_storage params;
        };

        //
        // show command
        //
        struct show : base {
            static constexpr string_ref name() { return "show"; }

            static constexpr inline char kind_param[] = "kind_param_name";
            static constexpr inline char name_param[] = "name_param";

            using command_params = util::type_list<
                named_param< kind_param, show_kind >,
                named_param< name_param, string_param >
            >;

            using params_storage = command_params::as_tuple;

            show(const params_storage &params) : params(params) {}
            show(params_storage &&params) : params(std::move(params)) {}

            void run(state_t &state) const override;

            params_storage params;
        };

        //
        // analyze command
        //
        struct analyze : base {
            static constexpr string_ref name() { return "analyze"; }

            static constexpr inline char analysis_param[] = "analysis_name";

            using command_params = util::type_list<
                named_param< analysis_param, analyze_func >
            >;

            using params_storage = command_params::as_tuple;

            explicit analyze(const params_storage &params) : params(params) {}
            explicit analyze(params_storage &&params) : params(std::move(params)) {}

            void run(state_t &state) const override;

            params_storage params;
        };

        //
        // meta command
        //
        struct meta : base {
            static constexpr string_ref name() { return "meta"; }

            static constexpr inline char action_param[] = "meta_action";
            static constexpr inline char symbol_param[]     = "symbol";
            static constexpr inline char identifier_param[] = "identifier";

            using command_params = util::type_list<
                named_param< action_param, meta_action >,
                named_param< identifier_param, integer_param >,
                named_param< symbol_param, string_param >
            >;

            using params_storage = command_params::as_tuple;

            meta(const params_storage &params) : params(params) {}
            meta(params_storage &&params) : params(std::move(params)) {}

            void run(state_t &state) const override;

            void add(state_t &state) const;
            void get(state_t &state) const;

            params_storage params;
        };

        //
        // raise command
        //
        struct raise : base {
            static constexpr string_ref name() { return "raise"; }

            static constexpr inline char pipeline_param[] = "pipeline_name";
            static constexpr inline char link_name_param[] = "link_name";

            using command_params =
                util::type_list< named_param< pipeline_param, string_param >,
                                 named_param< link_name_param, string_param > >;

            using params_storage = command_params::as_tuple;

            raise(const params_storage &params) : params(params) {}
            raise(params_storage &&params) : params(std::move(params)) {}

            void run(state_t &state) const override;

            params_storage params;
        };

        struct sticky : base {
            static constexpr string_ref name() { return "sticky"; }

            static constexpr inline char command_param[] = "command";

            using command_params =
                util::type_list< named_param< command_param, string_param > >;

            using params_storage = command_params::as_tuple;

            sticky(const params_storage &params) : params(params) {}
            sticky(params_storage &&params) : params(std::move(params)) {}

            void run(state_t &state) const override;

            params_storage params;
        };

        void add_sticky_command(string_ref cmd, state_t &state);

        using command_list = util::type_list< exit, help, load, show, analyze, meta, raise, sticky >;

    } // namespace cmd

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
            throw_error(("no match for param: " + tokens.front()).str().c_str());
        } else {
            using current_param = typename params_list::head;
            using rest          = typename params_list::tail;

            auto param = std::make_tuple(current_param::parse(tokens.front()));
            return std::tuple_cat(param, parse_params< rest >(tail(tokens)));
        }
    }

    template<>
    inline auto parse_params< cmd::sticky::command_params >(std::span< string_ref > tokens)
        -> typename cmd::sticky::command_params::as_tuple
    {
        return {cmd::sticky::command_params::head::parse(
            std::accumulate(tokens.begin(), tokens.end(), std::string{}, [] (std::string res, auto token) {
                return std::move(res) + " " + token.str();
            })
        )};
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
            return std::make_unique< cmd::help >();
        }

        if constexpr (commands::empty) {
            // we did not recursivelly match any of known commands
            throw_error(("no match for command: " + tokens.front()).str().c_str());
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

    command_ptr parse_command(std::span< command_token > tokens);

} // namespace vast::repl
