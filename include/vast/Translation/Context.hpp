// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <clang/AST/ASTContext.h>
#include <llvm/ADT/ScopedHashTable.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LogicalResult.h>
VAST_UNRELAX_WARNINGS

#include "vast/Dialect/HighLevel/HighLevelDialect.hpp"
#include "vast/Dialect/HighLevel/HighLevelOps.hpp"
#include "vast/Dialect/HighLevel/HighLevelTypes.hpp"
#include "vast/Util/Functions.hpp"

#include <variant>

namespace vast::hl
{
    using StringRef     = llvm::StringRef;
    using LogicalResult = mlir::LogicalResult;

    using Value       = mlir::Value;
    using Stmt        = mlir::Operation *;
    using ValueOrStmt = std::variant< mlir::Value, Stmt >;

    using AContext = clang::ASTContext;
    using MContext = mlir::MLIRContext;

    using ModuleRef = mlir::OwningModuleRef;

    struct TranslationContext {
        MContext &mctx;
        AContext &actx;
        ModuleRef &mod;

        dl::DataLayoutBlueprint dl;

        TranslationContext(MContext &mctx, AContext &actx, ModuleRef &mod)
            : mctx(mctx)
            , actx(actx)
            , mod(mod) {}

        // The symbol table maps a variable name to a value in the current scope.
        // Entering a function creates a new scope, and the function arguments are
        // added to the mapping. When the processing of a function is terminated, the
        // scope is destroyed and the mappings created in this scope are dropped.
        llvm::ScopedHashTable< StringRef, Value > symbols;

        LogicalResult declare(StringRef var, Value value) {
            if (symbols.count(var))
                return mlir::failure();
            symbols.insert(var, value);
            return mlir::success();
        }

        MContext &getMLIRContext() { return mctx; }
        AContext &getASTContext() { return actx; }
        ModuleRef &getModule() { return mod; }

        const dl::DataLayoutBlueprint &data_layout() const { return dl; }
        dl::DataLayoutBlueprint &data_layout() { return dl; }

        mlir::Region &getBodyRegion() { return mod->getBodyRegion(); }

        clang::SourceManager &getSourceManager() { return actx.getSourceManager(); }

        auto lookup_symbol(StringRef name) { return mod->lookupSymbol< mlir::FuncOp >(name); }

        auto lookup_typedecl(StringRef name) { return mod->lookupSymbol< TypeDeclOp >(name); }

        void emitError(llvm::Twine msg) { mod->emitError(msg); }
    };

    inline bool check(const ValueOrStmt &val) {
        return std::visit([](const auto &v) { return bool(v); }, val);
    }

    inline auto to_value = [](ValueOrStmt v) -> Value { return std::get< Value >(v); };

} // namespace vast::hl
