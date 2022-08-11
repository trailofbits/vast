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
#include "vast/Dialect/HighLevel/HighLevelAttributes.hpp"
#include "vast/Dialect/HighLevel/HighLevelTypes.hpp"
#include "vast/Util/Functions.hpp"
#include "vast/Util/ScopeTable.hpp"
#include "vast/Util/Common.hpp"

#include <variant>

namespace vast::hl
{
    struct CodeGenContext {
        MContext &mctx;
        AContext &actx;
        OwningModuleRef &mod;

        dl::DataLayoutBlueprint dl;

        CodeGenContext(MContext &mctx, AContext &actx, OwningModuleRef &mod)
            : mctx(mctx)
            , actx(actx)
            , mod(mod)
        {}

        using VarTable = ScopedValueTable< const clang::VarDecl*, Value >;

        VarTable vars;

        ScopedSymbolTable< mlir::FuncOp > functions;
        ScopedSymbolTable< TypeDefOp > type_defs;
        ScopedSymbolTable< TypeDeclOp > type_decls;

        using EnumDecls = ScopedValueTable< StringRef, EnumDeclOp >;
        EnumDecls enum_decls;

        using EnumConstants = ScopedValueTable< StringRef, EnumConstantOp >;
        EnumConstants enum_constants;

        size_t anonymous_count = 0;
        llvm::DenseMap< const clang::TagDecl *, std::string > tag_names;

        std::string get_decl_name(const clang::NamedDecl *decl) {
            if (decl->getIdentifier())
                return decl->getName().str();
            return "anonymous[" + std::to_string(decl->getID()) + "]";
        }

        std::string get_namespaced_for_decl_name(const clang::TagDecl *decl) {
            // gather contexts
            std::vector< const clang::DeclContext * > dctxs;
            for (const auto *dctx = decl->getDeclContext(); dctx; dctx = dctx->getParent()) {
                dctxs.push_back(dctx);
            }

            std::string name;
            for (const auto *dctx : llvm::reverse(dctxs)) {
                if (llvm::isa< clang::TranslationUnitDecl >(dctx))
                    continue;

                if (llvm::isa< clang::FunctionDecl >(dctx))
                    continue;

                if (const auto *d = llvm::dyn_cast< clang::TagDecl >(dctx)) {
                    name += get_decl_name(d);
                } else {
                    VAST_UNREACHABLE("unknown decl context: {0}", dctx->getDeclKindName());
                }

                name += "::";
            }

            return name;
        }

        std::string get_namespaced_decl_name(const clang::TagDecl *decl) {
            return get_namespaced_for_decl_name(decl) + get_decl_name(decl);
        }

        llvm::StringRef decl_name(const clang::TagDecl *decl) {
            if (tag_names.count(decl)) {
                return tag_names[decl];
            }

            auto name = get_namespaced_decl_name(decl);
            auto [it, _] = tag_names.try_emplace(decl, name);
            return it->second;
        }

        const dl::DataLayoutBlueprint &data_layout() const { return dl; }
        dl::DataLayoutBlueprint &data_layout() { return dl; }

        mlir::Region &getBodyRegion() { return mod->getBodyRegion(); }

        auto error(llvm::Twine msg) { return mod->emitError(msg); }

        template< typename Table, typename ValueType = typename Table::ValueType >
        ValueType symbol(Table &table, StringRef name, llvm::Twine msg, bool with_error = true) {
            if (auto val = table.lookup(name))
                return val;
            if (with_error)
                error(msg);
            return nullptr;
        }

        mlir::FuncOp lookup_function(StringRef name, bool with_error = true) {
            return symbol(functions, name, "error: undeclared function '" + name + "'", with_error);
        }

        TypeDeclOp lookup_typedecl(StringRef name, bool with_error = true) {
            return symbol(type_decls, name, "error: unknown type declaration '" + name + "'", with_error);
        }

        TypeDefOp lookup_typedef(StringRef name, bool with_error = true) {
            return symbol(type_defs, name, "error: unknown type definition '" + name + "'", with_error);
        }

        EnumDeclOp lookup_enum(StringRef name, bool with_error = true) {
            return symbol(enum_decls, name, "error: unknown enum '" + name + "'", with_error);
        }
    };
} // namespace vast::hl
