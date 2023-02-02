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
#include "mlir/Parser/Parser.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
VAST_UNRELAX_WARNINGS

#include "vast/Dialect/Dialects.hpp"
#include "vast/Dialect/HighLevel/HighLevelAttributes.hpp"
#include "vast/Dialect/HighLevel/HighLevelDialect.hpp"
#include "vast/Dialect/HighLevel/HighLevelOps.hpp"
#include "vast/Dialect/HighLevel/HighLevelTypes.hpp"
#include "vast/Dialect/HighLevel/Passes.hpp"
#include "vast/Util/Common.hpp"
#include "vast/Util/Symbols.hpp"

using memory_buffer  = std::unique_ptr< llvm::MemoryBuffer >;

namespace vast::cl
{
    namespace cl = llvm::cl;

    // clang-format off
    enum class show_symbol_type {
        none, function, type, record, var, global, all
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
                clEnumValN(show_symbol_type::var, "vars", "show variable symbols"),
                clEnumValN(show_symbol_type::global, "globs", "show global variable symbols"),
                clEnumValN(show_symbol_type::all, "all", "show all symbols")
            ),
            cl::init(show_symbol_type::none),
            cl::cat(queries)
        };
        cl::opt< std::string > show_symbol_users{ "symbol-users",
            cl::desc("Show users of a given symbol"),
            cl::value_desc("symbol name"),
            cl::init(""),
            cl::cat(queries)
        };
        cl::opt< std::string > scope_name{ "scope",
            cl::desc("Show values from scope of a given function"),
            cl::value_desc("function name"),
            cl::init(""),
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

    bool constrained_scope() { return !cl::options->scope_name.empty(); }

    template< typename... Ts >
    auto is_one_of() {
        return [](mlir::Operation *op) { return (mlir::isa< Ts >(op) || ...); };
    }

    template< typename T >
    auto is_global() {
        return [](mlir::Operation *op) {
            auto parent = op->getParentOp();
            return mlir::isa< T >(op) && is_one_of< mlir::ModuleOp, hl::TranslationUnitOp >()(parent);
        };
    }

    void show_value(auto value) {
        llvm::outs() << util::show_symbol_value(value) << "\n";
    }

    logical_result do_show_symbols(auto scope) {
        auto &show_kind = cl::options->show_symbols;

        auto show_if = [=](auto symbol, auto pred) {
            if (pred(symbol))
                show_value(symbol);
        };

        auto filter_kind = [=](cl::show_symbol_type kind) {
            return [=](auto symbol) {
                switch (kind) {
                    case cl::show_symbol_type::all: show_value(symbol); break;
                    case cl::show_symbol_type::type:
                        show_if(symbol, is_one_of< hl::TypeDefOp, hl::TypeDeclOp >());
                        break;
                    case cl::show_symbol_type::record:
                        show_if(symbol, is_one_of< hl::StructDeclOp >());
                        break;
                    case cl::show_symbol_type::var:
                        show_if(symbol, is_one_of< hl::VarDeclOp >());
                        break;
                    case cl::show_symbol_type::global:
                        show_if(symbol, is_global< hl::VarDeclOp >());
                        break;
                    case cl::show_symbol_type::function:
                        show_if(symbol, is_one_of< hl::FuncOp >());
                        break;
                    case cl::show_symbol_type::none: break;
                }
            };
        };

        util::symbols(scope, filter_kind(show_kind));
        return mlir::success();
    }

    logical_result do_show_users(auto scope) {
        auto &name = cl::options->show_symbol_users;
        util::yield_users(name.getValue(), scope, [](auto user) {
            user->print(llvm::outs());
            llvm::outs() << util::show_location(*user) << "\n";
        });

        return mlir::success();
    }
} // namespace vast::query

namespace vast
{
    logical_result get_scope_operation(auto parent, std::string_view scope_name, auto yield) {
        auto result =mlir::success();
        util::symbol_tables(parent, [&](mlir::Operation *op) {
            if (failed(yield(mlir::SymbolTable::lookupSymbolIn(op, scope_name)))) {
                result = mlir::failure();
            }
        });

        return result;
    }

    logical_result do_query(mcontext_t &ctx, memory_buffer buffer) {
        llvm::SourceMgr source_mgr;
        source_mgr.AddNewSourceBuffer(std::move(buffer), llvm::SMLoc());

        mlir::SourceMgrDiagnosticHandler manager_handler(source_mgr, &ctx);

        // Disable multi-threading when parsing the input file. This removes the
        // unnecessary/costly context synchronization when parsing.
        bool wasThreadingEnabled = ctx.isMultithreadingEnabled();
        ctx.disableMultithreading();

        owning_module_ref mod(mlir::parseSourceFile< vast_module >(source_mgr, &ctx));
        ctx.enableMultithreading(wasThreadingEnabled);
        if (!mod) {
            llvm::errs() << "error: cannot parse module\n";
            return mlir::failure();
        }

        auto process_scope = [&] (auto scope) {
            if (query::show_symbols()) {
                return query::do_show_symbols(scope);
            }

            if (query::show_symbol_users()) {
                return query::do_show_users(scope);
            }

            return mlir::success();
        };

        mlir::Operation *scope = mod.get();
        if (query::constrained_scope()) {
            return get_scope_operation(scope, cl::options->scope_name, process_scope);
        } else {
            return process_scope(scope);
        }
    }

    logical_result run(mcontext_t &ctx) {
        std::string err;
        if (auto input = mlir::openInputFile(cl::options->input_file, &err))
            return do_query(ctx, std::move(input));
        llvm::errs() << "error: " << err << "\n";
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

    vast::mcontext_t ctx(registry);
    ctx.loadAllAvailableDialects();

    std::exit(failed(vast::run(ctx)));
}
