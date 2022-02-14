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
#include "mlir/Support/MlirOptMain.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
VAST_UNRELAX_WARNINGS

#include "vast/Dialect/Dialects.hpp"
#include "vast/Dialect/HighLevel/Passes.hpp"

using context_t      = mlir::MLIRContext;
using logical_result = mlir::LogicalResult;

namespace cl = llvm::cl;

struct vast_query_options {
    cl::opt< std::string > input_file{ cl::Positional, cl::desc("<input file>"), cl::init("-") };
    cl::opt< bool > show_symbols{ "symbols", cl::desc("Show MLIR symbols"), cl::init(false) };
    cl::opt< std::string > show_symbol_users{ "symbol-users",
            cl::desc("Show users of a given symbol"),
            cl::value_desc("symbol name"), cl::init("-")
    };
};

static llvm::ManagedStatic< vast_query_options > options;

void register_options() { *options; }

static logical_result execute_query(context_t &ctx) { return mlir::success(); }
static logical_result run(context_t &ctx) {
    std::string err;
    if (auto input = mlir::openInputFile(options->input_file, &err)) {
        return execute_query(ctx, std::move(input));
    }

    llvm::errs() << err << "\n";
    return mlir::failure();
}

int main(int argc, char **argv) {
    register_options();
    cl::ParseCommandLineOptions(argc, argv, "VAST source querying tool\n");

    mlir::DialectRegistry registry;
    vast::registerAllDialects(registry);
    mlir::registerAllDialects(registry);
    context_t context(registry);
    context.loadAllAvailableDialects();

    std::exit(failed(run(context)));
}
