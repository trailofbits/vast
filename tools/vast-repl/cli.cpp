// Copyright (c) 2023, Trail of Bits, Inc.

#include "vast/repl/cli.hpp"
#include "vast/repl/config.hpp"

namespace vast::repl {

    void cli_t::initialize(std::filesystem::path config_path) {
        std::ifstream input(config_path);
        if (!input.is_open()) {
            llvm::errs() << "error: failed to opean config file\n";
        }

        auto cfg = ini::config::parse(input);

        for (auto sec: cfg.sections) {
            // setup initially sticky commands
            // e.g., one cas set deafault printing of module
            // putting `show module` in the config sticky section
            if (sec.is_sticky_section()) {
                for (auto cmd : sec.content) {
                    cmd::add_sticky_command(cmd, state);
                }
            }

            if (sec.is_pipeline_section()) {
                auto name = sec.last_name();
                state.pipelines[name.str()] = pipeline{
                    .passes = sec.content
                };
            }
        }
    }

} // namespace vast::repl
