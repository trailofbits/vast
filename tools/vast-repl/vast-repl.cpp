// Copyright (c) 2022-present, Trail of Bits, Inc.

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/MlirOptMain.h"
#include "vast/repl/linenoise.hpp"
VAST_UNRELAX_WARNINGS

#include "vast/Dialect/Dialects.hpp"
#include "vast/Dialect/HighLevel/Passes.hpp"
#include "vast/Util/Common.hpp"
#include "vast/repl/cli.hpp"
#include "vast/repl/command.hpp"

using logical_result = mlir::LogicalResult;

namespace vast::repl
{
    struct prompt {
        explicit prompt(MContext &ctx)
            : ctx(ctx) {}

        logical_result run() {
            const auto path = ".vast-repl.history";

            linenoise::SetHistoryMaxLen(1000);
            linenoise::LoadHistory(path);

            cli_t interp;

            llvm::outs() << "Welcome to 'vast-repl', an interactive MLIR modifier. Type 'help' to "
                            "get started.\n";

            while (!interp.exit()) {
                std::string cmd;
                if (auto quit = linenoise::Readline("> ", cmd)) {
                    break;
                }

                if (failed(interp.command(cmd))) {
                    return mlir::failure();
                }

                linenoise::AddHistory(cmd.c_str());
                linenoise::SaveHistory(path);
            }

            return mlir::success();
        }

        MContext &ctx;
    };

} // namespace vast::repl

int main(int argc, char **argv) {
    mlir::DialectRegistry registry;
    vast::registerAllDialects(registry);
    mlir::registerAllDialects(registry);

    vast::MContext ctx(registry);
    ctx.loadAllAvailableDialects();

    std::exit(failed(vast::repl::prompt(ctx).run()));
}
