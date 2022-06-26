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
#include "vast/Dialect/HighLevel/HighLevelAttributes.hpp"
#include "vast/Dialect/HighLevel/HighLevelDialect.hpp"
#include "vast/Dialect/HighLevel/HighLevelOps.hpp"
#include "vast/Dialect/HighLevel/HighLevelTypes.hpp"
#include "vast/Dialect/HighLevel/Passes.hpp"
#include "vast/Util/Common.hpp"
#include "vast/Util/Symbols.hpp"

using memory_buffer  = std::unique_ptr< llvm::MemoryBuffer >;
using logical_result = mlir::LogicalResult;

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
    using vast_symbol_interface = vast::VastSymbolOpInterface;
    using mlir_symbol_interface = mlir::SymbolOpInterface;

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
            return mlir::isa< T >(op) && mlir::isa< mlir::ModuleOp >(op->getParentOp());
        };
    }

    auto show_name(vast_symbol_interface value) { return value.getSymbolName(); }

    auto show_name(mlir_symbol_interface value) { return value.getName(); }

    void show_location(auto &value) {
        auto loc = value.getLoc();
        if (auto file_loc = loc.template dyn_cast< mlir::FileLineColLoc >()) {
            llvm::outs() << " : " << file_loc.getFilename().getValue() << ":" << file_loc.getLine()
                         << ":" << file_loc.getColumn();
        } else {
            llvm::outs() << " : " << loc;
        }
    }

    void show_value(auto value) {
        llvm::outs() << value->getName() << " : " << show_name(value);
        show_location(value);
        llvm::outs() << "\n";
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
                        show_if(symbol, is_one_of< hl::RecordDeclOp >());
                        break;
                    case cl::show_symbol_type::var:
                        show_if(symbol, is_one_of< hl::VarDecl >());
                        break;
                    case cl::show_symbol_type::global:
                        show_if(symbol, is_global< hl::VarDecl >());
                        break;
                    case cl::show_symbol_type::function:
                        show_if(symbol, is_one_of< mlir::FuncOp >());
                        break;
                    case cl::show_symbol_type::none: break;
                }
            };
        };

        util::symbols(scope, filter_kind(show_kind));
        return mlir::success();
    }

    using maybe_range = llvm::Optional< std::vector< mlir::Operation > >;

    void yield_symbol_users(vast_symbol_interface op, auto scope, auto yield) {
        for (auto user : op->getUsers()) {
            yield(user);
        }
    };

    void yield_symbol_users(mlir_symbol_interface op, auto scope, auto yield) {
        if (auto users = op.getSymbolUses(scope)) {
            for (auto use : users.getValue()) {
                yield(use.getUser());
            }
        }
    };

    void yield_users(llvm::StringRef symbol, auto scope, auto yield) {
        auto filter_symbols = [&](auto op) {
            if (show_name(op) == symbol) {
                yield_symbol_users(op, scope, yield);
            }
        };

        util::symbols(scope, filter_symbols);
    }

    logical_result do_show_users(auto scope) {
        auto &name = cl::options->show_symbol_users;
        yield_users(name.getValue(), scope, [](auto user) {
            user->print(llvm::outs());
            show_location(*user);
            llvm::outs() << "\n";
        });

        return mlir::success();
    }
} // namespace vast::query

namespace vast
{
    logical_result get_scope_operation(auto parent, std::string_view scope_name, auto yield) {
        auto result = parent->walk([&](mlir::Operation *op) {
            if (mlir::isa< hl::TranslationUnitOp >(op)) {
                if (failed(yield(mlir::SymbolTable::lookupSymbolIn(op, scope_name)))) {
                    return mlir::WalkResult::interrupt();
                }
            }
            return mlir::WalkResult::advance();
        });

        return result.wasInterrupted() ? mlir::failure() : mlir::success();
    }

    logical_result do_query(MContext &ctx, memory_buffer buffer) {
        llvm::SourceMgr source_mgr;
        source_mgr.AddNewSourceBuffer(std::move(buffer), llvm::SMLoc());

        mlir::SourceMgrDiagnosticHandler manager_handler(source_mgr, &ctx);

        // Disable multi-threading when parsing the input file. This removes the
        // unnecessary/costly context synchronization when parsing.
        bool wasThreadingEnabled = ctx.isMultithreadingEnabled();
        ctx.disableMultithreading();

        OwningModuleRef mod(mlir::parseSourceFile(source_mgr, &ctx));
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

    logical_result run(MContext &ctx) {
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

    vast::MContext ctx(registry);
    ctx.loadAllAvailableDialects();

    std::exit(failed(vast::run(ctx)));
}
