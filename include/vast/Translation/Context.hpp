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
#include "vast/Util/ScopeTable.hpp"
#include "vast/Util/Common.hpp"

#include <variant>

namespace vast::hl
{

    struct TranslationContext {
        MContext &mctx;
        AContext &actx;
        OwningModuleRef &mod;

        dl::DataLayoutBlueprint dl;

        TranslationContext(MContext &mctx, AContext &actx, OwningModuleRef &mod)
            : mctx(mctx)
            , actx(actx)
            , mod(mod) {}

        using VarTable = ScopedValueTable< clang::VarDecl*, Value >;

        VarTable vars;

        ScopedSymbolTable< mlir::FuncOp > functions;
        ScopedSymbolTable< TypeDefOp > type_defs;
        ScopedSymbolTable< TypeDeclOp > type_decls;

        using EnumDecls = ScopedValueTable< StringRef, EnumDeclOp >;
        EnumDecls enum_decls;

        using EnumConstants = ScopedValueTable< StringRef, EnumConstantOp >;
        EnumConstants enum_constants;

        size_t anonymous_count = 0;
        llvm::DenseMap< clang::TagDecl *, std::string > tag_names;

        llvm::StringRef elaborated_name(clang::TagDecl *decl) {
            if (tag_names.count(decl)) {
                return tag_names[decl];
            }

            std::string name = decl->getKindName().str() + " ";
            if (decl->getIdentifier()) {
                name += decl->getName().str();
            } else {
                name += "anonymous." + std::to_string(anonymous_count++);
            }

            auto [it, _] = tag_names.try_emplace(decl, name);
            return it->second;
        }

        MContext &getMLIRContext() { return mctx; }
        AContext &getASTContext() { return actx; }
        OwningModuleRef &getModule() { return mod; }

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

        TypeDeclOp lookup_typedecl(StringRef name) {
            return symbol(type_decls, name, "error: unknown type declaration '" + name + "'");
        }

        TypeDefOp lookup_typedef(StringRef name) {
            return symbol(type_defs, name, "error: unknown type definition '" + name + "'");
        }

        EnumDeclOp lookup_enum(StringRef name) {
            return symbol(enum_decls, name, "error: unknown enum '" + name + "'");
        }
    };

    inline bool check(const ValueOrStmt &val) {
        return std::visit([](const auto &v) { return bool(v); }, val);
    }

    inline auto to_value = [](ValueOrStmt v) -> Value { return std::get< Value >(v); };

} // namespace vast::hl
