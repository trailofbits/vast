// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/repl/command.hpp"
#include "vast/repl/common.hpp"
#include "vast/repl/state.hpp"

#include <exception>
#include <memory>

namespace vast::repl
{
    struct cli_t {
        explicit cli_t(mcontext_t &ctx) : state(ctx) {}

        std::string_view help() { return "cli help"; }

        bool exit() const { return state.exit; }

        void exec(std::string_view line) try {
            auto tokens = parse_tokens(line);
            exec(parse_command(tokens));
        } catch (std::exception &e) {
            llvm::errs() << "error: " << e.what() << '\n';
        }

        void exec(command_ptr cmd) try {
            cmd->run(state);
        } catch (std::exception &e) {
            llvm::errs() << "error: " << e.what() << '\n';
        }

      private:
        state_t state;
    };

} // namespace vast::repl
