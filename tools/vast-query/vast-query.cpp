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

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
VAST_UNRELAX_WARNINGS

#include "vast/Dialect/Dialects.hpp"
#include "vast/Dialect/HighLevel/Passes.hpp"

using context_t      = mlir::MLIRContext;
using mmodule_t      = mlir::OwningModuleRef;
using memory_buffer  = std::unique_ptr< llvm::MemoryBuffer >;
using logical_result = mlir::LogicalResult;

namespace cl = llvm::cl;

enum query_type
{
    symbols,
    symbol_users
};

struct vast_query_options {
    // clang-format off
    cl::opt< std::string > input_file{ cl::desc("<input file>"),
        cl::Positional, cl::init("-")
    };
    cl::opt< bool > show_symbols{ "symbols",
        cl::desc("Show MLIR symbols"), cl::init(false)
    };
    cl::opt< std::string > show_symbol_users{ "symbol-users",
        cl::desc("Show users of a given symbol"),
        cl::value_desc("symbol name"), cl::init("-")
    };
    // clang-format on
};

static llvm::ManagedStatic< vast_query_options > options;

void register_options() { *options; }

logical_result show_symbols(const mmodule_t &mod) { return mlir::success(); }

logical_result show_symbol_users(const mmodule_t &mod, std::string_view symbol) {
    return mlir::success();
}

logical_result query(context_t &ctx, memory_buffer buffer) {
    llvm::SourceMgr source_mgr;
    source_mgr.AddNewSourceBuffer(std::move(buffer), llvm::SMLoc());

    mlir::SourceMgrDiagnosticHandler manager_handler(source_mgr, &ctx);

    // Disable multi-threading when parsing the input file. This removes the
    // unnecessary/costly context synchronization when parsing.
    bool wasThreadingEnabled = ctx.isMultithreadingEnabled();
    ctx.disableMultithreading();

    mlir::OwningModuleRef mod(mlir::parseSourceFile(source_mgr, &ctx));
    ctx.enableMultithreading(wasThreadingEnabled);

    if (!mod)
        return mlir::failure();

    if (options->show_symbols) {
        return show_symbols(mod);
    }

    if (!options->show_symbol_users.empty()) {
        return show_symbol_users(mod, options->show_symbol_users);
        // return symbol users
    }

    return mlir::success();
}

static logical_result run(context_t &ctx) {
    std::string err;
    if (auto input = mlir::openInputFile(options->input_file, &err))
        return query(ctx, std::move(input));
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
