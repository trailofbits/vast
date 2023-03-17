// Copyright (c) 2022-present, Trail of Bits, Inc.

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "vast/repl/linenoise.hpp"
VAST_UNRELAX_WARNINGS

#include "vast/Dialect/Dialects.hpp"
#include "vast/Dialect/HighLevel/Passes.hpp"
#include "vast/Util/Common.hpp"
#include "vast/repl/cli.hpp"
#include "vast/repl/command.hpp"

using args_t = std::vector< vast::string_ref >;

args_t load_args(int argc, char **argv) {
    args_t args;

    for (int i = 1; i < argc; i++) {
        args.push_back(argv[i]);
    }

    return args;
}

namespace vast::repl
{
    struct prompt {
        explicit prompt(mcontext_t &ctx) : cli(ctx) {}

        void init(std::span< string_ref > args) {
            if (args.size() == 1) {
                auto params = parse_params< cmd::load::command_params >(args);
                cli.exec(make_command< cmd::load >(params));
            } else {
                VAST_UNREACHABLE("unsupported arguments");
            }
        }

        logical_result run() try {
            const auto path = ".vast-repl.history";

            linenoise::SetHistoryMaxLen(1000);
            linenoise::LoadHistory(path);

            llvm::outs() << "Welcome to 'vast-repl', an interactive MLIR modifier. Type 'help' to "
                            "get started.\n";

            while (!cli.exit()) {
                std::string cmd;
                if (auto quit = linenoise::Readline("> ", cmd)) {
                    break;
                }

                cli.exec(cmd);

                linenoise::AddHistory(cmd.c_str());
                linenoise::SaveHistory(path);
            }

            return mlir::success();
        } catch (std::exception &e) {
            llvm::errs() << "error: " << e.what() << '\n';
            return mlir::failure();
        }

        cli_t cli;
    };

} // namespace vast::repl

int main(int argc, char **argv) try {
    mlir::DialectRegistry registry;
    vast::registerAllDialects(registry);
    mlir::registerAllDialects(registry);

    args_t args = load_args(argc, argv);

    vast::mcontext_t ctx(registry);
    ctx.loadAllAvailableDialects();

    auto prompt = vast::repl::prompt(ctx);

    if (!args.empty()) {
        prompt.init(args);
    }

    std::exit(failed(prompt.run()));

} catch (std::exception &e) {
    llvm::errs() << "error: " << e.what() << '\n';
    std::exit(1);
}
