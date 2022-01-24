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

    template< typename T >
    struct ScopedSymbolTable : llvm::ScopedHashTable< StringRef, T > {
        using ValueType = T;

        using Base = llvm::ScopedHashTable< StringRef, T >;
        using Base::Base;

        LogicalResult declare(StringRef name, T value) {
            if (this->count(name))
                return mlir::failure();
            this->insert(name, value);
            return mlir::success();
        }
    };

    struct TranslationContext {
        MContext &mctx;
        AContext &actx;
        ModuleRef &mod;

        dl::DataLayoutBlueprint dl;

        TranslationContext(MContext &mctx, AContext &actx, ModuleRef &mod)
            : mctx(mctx)
            , actx(actx)
            , mod(mod) {}

        ScopedSymbolTable< Value > variables;
        ScopedSymbolTable< mlir::FuncOp > functions;
        ScopedSymbolTable< TypeDefOp > type_defs;
        ScopedSymbolTable< TypeDeclOp > type_decls;

        MContext &getMLIRContext() { return mctx; }
        AContext &getASTContext() { return actx; }
        ModuleRef &getModule() { return mod; }

        const dl::DataLayoutBlueprint &data_layout() const { return dl; }
        dl::DataLayoutBlueprint &data_layout() { return dl; }

        mlir::Region &getBodyRegion() { return mod->getBodyRegion(); }

        clang::SourceManager &getSourceManager() { return actx.getSourceManager(); }

        auto error(llvm::Twine msg) { return mod->emitError(msg); }

        template< typename Table, typename ValueType = typename Table::ValueType >
        ValueType symbol(Table &table, StringRef name, llvm::Twine msg) {
            if (auto val = table.lookup(name))
                return val;
            error(msg);
            return nullptr;
        }

        mlir::FuncOp lookup_function(StringRef name) {
            return symbol(functions, name, "error: undeclared function '" + name + "'");
        }

        Value lookup_variable(StringRef name) {
            return symbol(variables, name, "error: undeclared variable '" + name + "'");
        }

        TypeDeclOp lookup_typedecl(StringRef name) {
            return symbol(type_decls, name, "error: unknown type declaration '" + name + "'");
        }

        TypeDefOp lookup_typedef(StringRef name) {
            return symbol(type_defs, name, "error: unknown type definition '" + name + "'");
        }
    };

    inline bool check(const ValueOrStmt &val) {
        return std::visit([](const auto &v) { return bool(v); }, val);
    }

    inline auto to_value = [](ValueOrStmt v) -> Value { return std::get< Value >(v); };

} // namespace vast::hl
