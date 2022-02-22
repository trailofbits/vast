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
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
VAST_UNRELAX_WARNINGS

#include "vast/Dialect/Dialects.hpp"
#include "vast/Dialect/HighLevel/Passes.hpp"

using context_t      = mlir::MLIRContext;
using module_t       = mlir::OwningModuleRef;
using memory_buffer  = std::unique_ptr< llvm::MemoryBuffer >;
using logical_result = mlir::LogicalResult;

namespace vast::cl
{
    namespace cl = llvm::cl;

    // clang-format off
    enum class show_symbol_type {
        none, function, type, record, all
    };

    cl::OptionCategory generic("Vast Generic Options");
    cl::OptionCategory queries("Vast Queries Options");

    struct vast_query_options {
        cl::opt< std::string > input_file{
            cl::desc("<input file>"),
            cl::Positional,
            cl::init("-"),
            cl::cat(generic)
        };
        cl::opt< show_symbol_type > show_symbols{ "show-symbols",
            cl::desc("Show MLIR symbols"),
            cl::values(
                clEnumValN(show_symbol_type::function, "functions", "show function symbols"),
                clEnumValN(show_symbol_type::type, "types", "show type symbols"),
                clEnumValN(show_symbol_type::record, "records", "show record symbols"),
                clEnumValN(show_symbol_type::all, "all", "show all symbols")
            ),
            cl::init(show_symbol_type::none),
            cl::cat(queries)
        };
        cl::opt< std::string > show_symbol_users{ "symbol-users",
            cl::desc("Show users of a given symbol"),
            cl::value_desc("symbol name"),
            cl::init("-"),
            cl::cat(queries)
        };
    };
    // clang-format on

    static llvm::ManagedStatic< vast_query_options > options;

    void register_options() { *options; }
} // namespace vast::cl

namespace vast::query
{
    bool show_symbols() { return cl::options->show_symbols != cl::show_symbol_type::none; }

    bool show_symbol_users() { return !cl::options->show_symbol_users.empty(); }

    // generator< symbol > symbols(module_t &mod) { ... }

    //

    logical_result do_show_symbols(module_t &mod) {
        // auto show_kind = cl::options->show_symbols;
        mod->dump();
        return mlir::success();
    }

    logical_result do_show_users(module_t &mod) {
        // auto symbol = cl::options->show_symbols;
        return mlir::success();
    }
} // namespace vast::query

namespace vast
{
    logical_result do_query(context_t &ctx, memory_buffer buffer) {
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

        if (query::show_symbols()) {
            return query::do_show_symbols(mod);
        }

        if (query::show_symbol_users()) {
            return query::do_show_users(mod);
        }

        return mlir::success();
    }

    logical_result run(context_t &ctx) {
        std::string err;
        if (auto input = mlir::openInputFile(cl::options->input_file, &err))
            return do_query(ctx, std::move(input));
        llvm::errs() << err << "\n";
        return mlir::failure();
    }

} // namespace vast

int main(int argc, char **argv) {
    llvm::cl::HideUnrelatedOptions({ &vast::cl::generic, &vast::cl::queries });
    vast::cl::register_options();
    llvm::cl::ParseCommandLineOptions(argc, argv, "VAST source querying tool\n");

    mlir::DialectRegistry registry;
    vast::registerAllDialects(registry);
    mlir::registerAllDialects(registry);

    context_t ctx(registry);
    ctx.loadAllAvailableDialects();

    std::exit(failed(vast::run(ctx)));
}
