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
#include "vast/Dialect/Core/CoreOps.hpp"
#include "vast/Dialect/Core/SymbolTable.hpp"
#include "vast/Dialect/HighLevel/HighLevelAttributes.hpp"
#include "vast/Dialect/HighLevel/HighLevelDialect.hpp"
#include "vast/Dialect/HighLevel/HighLevelOps.hpp"
#include "vast/Dialect/HighLevel/HighLevelTypes.hpp"
#include "vast/Dialect/HighLevel/Passes.hpp"

#include "vast/Dialect/Core/Interfaces/SymbolInterface.hpp"
#include "vast/Dialect/Core/Interfaces/SymbolTableInterface.hpp"

#include "vast/Util/Common.hpp"

using memory_buffer  = std::unique_ptr< llvm::MemoryBuffer >;

namespace vast::cl
{
    namespace cl = llvm::cl;

    // clang-format off
    enum class show_symbol_type {
        none, function, var, type, all
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
                clEnumValN(show_symbol_type::var, "vars", "show variable symbols"),
                clEnumValN(show_symbol_type::type, "types", "show type symbols"),
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
        return [](operation op) { return (mlir::isa< Ts >(op) || ...); };
    }

    template< typename T >
    auto is_global() {
        return [](operation op) {
            auto parent = op->getParentOp();
            return mlir::isa< T >(op) && is_one_of< core::ModuleOp, hl::TranslationUnitOp >()(parent);
        };
    }

    std::string show_location(auto &value) {
        auto loc = value.getLoc();
        std::string buff;
        llvm::raw_string_ostream ss(buff);
        if (auto file_loc = mlir::dyn_cast< mlir::FileLineColLoc >(loc)) {
            ss << " : " << file_loc.getFilename().getValue()
               << ":"   << file_loc.getLine()
               << ":"   << file_loc.getColumn();
        } else {
            ss << " : " << loc;
        }

        return ss.str();
    }

    std::string show_symbol_value(auto value) {
        std::string buff;
        llvm::raw_string_ostream ss(buff);
        ss << value->getName() << " : " << value.getSymbolName() << " " << show_location(value);
        return ss.str();
    }

    void show_value(auto value) {
        llvm::outs() << show_symbol_value(value) << "\n";
    }

    logical_result do_show_symbols(auto scope) {
        auto show = [&] (auto symbol) { show_value(symbol); };

        switch (cl::options->show_symbols) {
            case cl::show_symbol_type::function:
                core::symbols< core::func_symbol >(scope, show);
                break;
            case cl::show_symbol_type::var:
                core::symbols< core::var_symbol >(scope, show);
                break;
            case cl::show_symbol_type::type:
                core::symbols< core::type_symbol >(scope, show);
                break;
            case cl::show_symbol_type::all:
                core::symbols< core::symbol >(scope, show);
                break;
            case cl::show_symbol_type::none: break;
        }

        return mlir::success();
    }

    logical_result do_show_users(auto scope) {
        auto &name = cl::options->show_symbol_users;

        auto show_users = [scope] (operation decl) {
            for (auto use : core::symbol_table::get_symbol_uses(decl, scope)) {
                auto user = use.getUser();
                user->print(llvm::outs());
                llvm::outs() << show_location(*user) << "\n";
            }
        };

        // TODO: walk decl above the scope
        core::symbols< core::symbol >(scope, [&] (auto decl) {
            if (decl.getSymbolName() == name) {
                show_value(decl);
                show_users(decl);
            }
        });


        return mlir::success();
    }
} // namespace vast::query

namespace vast
{
    logical_result process_named_scope(auto root, string_ref scope_name, auto &&process) {
        auto result = mlir::success();
        core::symbol_tables(root, [&] (operation op) {
            if (auto symbol = mlir::dyn_cast< core::symbol >(op)) {
                if (symbol.getSymbolName() == scope_name) {
                    if (mlir::failed(process(op))) {
                        result = mlir::failure();
                    }
                }
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

        owning_mlir_module_ref mod(mlir::parseSourceFile< mlir_module >(source_mgr, &ctx));
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

        operation scope = mod.get();
        if (query::constrained_scope()) {
            return process_named_scope(scope, cl::options->scope_name, process_scope);
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
