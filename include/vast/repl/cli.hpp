// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/repl/command.hpp"
#include "vast/repl/state.hpp"

#include <exception>
#include <memory>

namespace vast::repl
{
    struct cli_t {
        std::string_view help() { return "cli help"; }

        bool exit() const { return state.exit; }

        logical_result exec(std::string_view line) try {
            auto tokens = parse_tokens(line);
            return exec(parse_command(tokens));
        } catch (std::exception &e) {
            llvm::errs() << "error: " << e.what() << '\n';
            return mlir::failure();
        }

        logical_result exec(command_ptr cmd) try {
            cmd->run(state);
            return mlir::success();
        } catch (std::exception &e) {
            llvm::errs() << "error: " << e.what() << '\n';
            return mlir::failure();
        }


      private:
        state_t state;
    };

} // namespace vast::repl
