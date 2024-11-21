// Copyright (c) 2021-present, Trail of Bits, Inc.

#include <string>

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include <llvm/Support/FileSystem.h>
#include <llvm/Support/raw_ostream.h>
VAST_UNRELAX_WARNINGS

#include "vast/Conversion/Parser/Passes.hpp"
#include "vast/Dialect/Dialects.hpp"

#include "vast/Dialect/Parser/Dialect.hpp"

#include "SarifPasses.hpp"

namespace vast {

#ifdef VAST_ENABLE_SARIF
    struct SarifWriter : mlir::PassWrapper< SarifWriter, mlir::OperationPass< mlir::ModuleOp > >
    {
        std::vector< gap::sarif::result > results;
        std::string path;

        SarifWriter(std::string path) : path(path) {}

        void runOnOperation() override {}

        ~SarifWriter() {
            std::error_code ec;
            llvm::raw_fd_ostream os(path, ec, llvm::sys::fs::OF_None);
            if (ec) {
                VAST_FATAL("Failed to open file for SARIF output: {}", ec.message());
            }
            gap::sarif::root root{
                .version = gap::sarif::version::k2_1_0,
                .runs{
                      {
                        {
                            .tool{
                                .driver{
                                    .name{ "detect-parsers" },
                                },
                            },
                            .results{ results },
                        },
                    }, },
            };

            nlohmann::json root_json = root;

            os << root_json.dump(2);
        }
    };

    struct SarifOptions : mlir::PassPipelineOptions< SarifOptions >
    {
        Option< std::string > out_path{ *this, "output",
                                        llvm::cl::desc("Output SARIF file path.") };
    };

    void registerSarifPasses() {
        mlir::PassPipelineRegistration< SarifOptions >(
            "parser-source-to-sarif", "Dumps all pr.source locations to a SARIF file.",
            [](mlir::OpPassManager &pm, const SarifOptions &opts) {
                auto writer = std::make_unique< SarifWriter >(opts.out_path);
                pm.addPass(std::make_unique< vast::ParserSourceDetector >(writer->results));
                pm.addPass(std::move(writer));
            }
        );
    }
#else

    void registerSarifPasses() {}
#endif
} // namespace vast

int main(int argc, char **argv) {
    mlir::DialectRegistry registry;
    // register dialects
    vast::registerAllDialects(registry);
    mlir::registerAllDialects(registry);

    vast::registerParserConversionPasses();
    vast::registerSarifPasses();
    registry.insert< vast::pr::ParserDialect >();

    return mlir::asMainReturnCode(
        mlir::MlirOptMain(argc, argv, "VAST Parser Detection driver\n", registry)
    );
}
