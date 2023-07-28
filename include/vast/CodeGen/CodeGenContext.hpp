// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <clang/AST/GlobalDecl.h>
#include <clang/AST/ASTContext.h>
#include <llvm/ADT/ScopedHashTable.h>
#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/IR/GlobalValue.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LogicalResult.h>
VAST_UNRELAX_WARNINGS

#include "vast/CodeGen/CodeGenScope.hpp"
#include "vast/CodeGen/ScopeContext.hpp"
#include "vast/CodeGen/Mangler.hpp"

#include "vast/Dialect/HighLevel/HighLevelDialect.hpp"
#include "vast/Dialect/HighLevel/HighLevelOps.hpp"
#include "vast/Dialect/HighLevel/HighLevelAttributes.hpp"
#include "vast/Dialect/HighLevel/HighLevelTypes.hpp"
#include "vast/Util/Functions.hpp"
#include "vast/Util/Common.hpp"
#include "vast/Util/Triple.hpp"

#include <variant>

namespace vast::cg
{
    namespace detail {
        static inline owning_module_ref create_module(mcontext_t &mctx, acontext_t &actx) {
            // TODO(Heno): fix module location
            auto module_ref = owning_module_ref(vast_module::create(mlir::UnknownLoc::get(&mctx)));
            // TODO(cg): For now we do not have our own operation, so we cannot
            //           introduce new ctor.
            set_triple(*module_ref, actx.getTargetInfo().getTriple().str());
            return module_ref;
        }
    } // namespace detail

    struct CodeGenContext {
        mcontext_t &mctx;
        acontext_t &actx;
        owning_module_ref mod;

        dl::DataLayoutBlueprint dl;

        CodeGenContext(mcontext_t &mctx, acontext_t &actx, owning_module_ref &&mod)
            : mctx(mctx)
            , actx(actx)
            , mod(std::move(mod))
            , mangler(actx.createMangleContext())
        {}

        CodeGenContext(mcontext_t &mctx, acontext_t &actx)
            : CodeGenContext(mctx, actx, detail::create_module(mctx, actx))
        {}

        lexical_scope_context *current_lexical_scope = nullptr;

        // Never move this!
        // It owns the strings that mangled_name_ref uses
        CodeGenMangler mangler;

        using VarTable = scoped_table< const clang::VarDecl *, Value >;
        VarTable vars;

        using TypeDefTable = scoped_table< const clang::TypedefDecl *, hl::TypeDefOp >;
        TypeDefTable typedefs;

        using TypeDeclTable = scoped_table< const clang::TypeDecl *, hl::TypeDeclOp >;
        TypeDeclTable typedecls;

        using FuncDeclTable = scoped_table< mangled_name_ref, hl::FuncOp >;
        FuncDeclTable funcdecls;

        using EnumDecls = scoped_table< const clang::EnumDecl *, hl::EnumDeclOp >;
        EnumDecls enumdecls;

        using EnumConstants = scoped_table< const clang::EnumConstantDecl *, hl::EnumConstantOp >;
        EnumConstants enumconsts;

        using LabelTable = scoped_table< const clang::LabelDecl*, hl::LabelDeclOp >;
        LabelTable labels;

        size_t anonymous_count = 0;
        llvm::DenseMap< const clang::NamedDecl *, std::string > tag_names;

        /// Set of global decls for which we already diagnosed mangled name conflict.
        /// Required to not issue a warning (on a mangling conflict) multiple times
        /// for the same decl.
        llvm::DenseSet< clang::GlobalDecl > diagnosed_conflicting_definitions;

        llvm::DenseMap< mangled_name_ref, mlir_value > global_decls;

        // A set of references that have only been seen via a weakref so far. This is
        // used to remove the weak of the reference if we ever see a direct reference
        // or a definition.
        llvm::SmallPtrSet< operation, 10 > weak_ref_references;

        // This contains all the decls which have definitions but which are deferred
        // for emission and therefore should only be output if they are actually
        // used. If a decl is in this, then it is known to have not been referenced
        // yet.
        std::map< mangled_name_ref, clang::GlobalDecl > deferred_decls;

        // A queue of (optional) vtables to consider emitting.
        std::vector< const clang::CXXRecordDecl * > deferred_vtables;

        mangled_name_ref get_mangled_name(clang::GlobalDecl decl) {
            return mangler.get_mangled_name(decl, actx.getTargetInfo(), /* module name hash */ "");
        }

        // This is a list of deferred decls which we have seen that *are* actually
        // referenced. These get code generated when the module is done.
        std::vector< clang::GlobalDecl > deferred_decls_to_emit;
        void add_deferred_decl_to_emit(clang::GlobalDecl decl) {
            deferred_decls_to_emit.emplace_back(decl);
        }

        // After HandleTranslation finishes, differently from deferred_decls_to_emit,
        // default_methods_to_emit is only called after a set of vast passes run.
        // See add_default_methods_to_emit usage for examples.
        std::vector< clang::GlobalDecl > default_methods_to_emit;
        void add_default_methods_to_emit(clang::GlobalDecl decl) {
            default_methods_to_emit.emplace_back(decl);
        }

        operation get_global_value(mangled_name_ref name) {
            if (auto global = mlir::SymbolTable::lookupSymbolIn(mod.get(), name.name))
                return global;
            return {};
        }

        mlir_value get_global_value(const clang::Decl * /* decl */) {
            VAST_UNIMPLEMENTED;
        }

        std::string get_decl_name(const clang::NamedDecl *decl) {
            if (decl->getIdentifier())
                return decl->getName().str();
            return "anonymous[" + std::to_string(decl->getID()) + "]";
        }

        std::string get_namespaced_for_decl_name(const clang::NamedDecl *decl) {
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

                if (llvm::isa< clang::LinkageSpecDecl >(dctx))
                    continue;

                if (const auto *d = llvm::dyn_cast< clang::NamedDecl >(dctx)) {
                    name += get_decl_name(d);
                } else {
                    VAST_UNREACHABLE("unknown decl context: {0}", dctx->getDeclKindName());
                }

                name += "::";
            }

            return name;
        }

        std::string get_namespaced_decl_name(const clang::NamedDecl *decl) {
            return get_namespaced_for_decl_name(decl) + get_decl_name(decl);
        }

        llvm::StringRef decl_name(const clang::NamedDecl *decl) {
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

        template< typename Table, typename Token, typename ValueType = typename Table::value_type >
        ValueType symbol(Table &table, const Token &token, llvm::Twine msg, bool with_error = true) {
            if (auto val = table.lookup(token))
                return val;
            if (with_error)
                error(msg);
            return nullptr;
        }

        hl::FuncOp lookup_function(mangled_name_ref mangled, bool with_error = true) {
            return symbol(funcdecls, mangled, "undeclared function '" + mangled.name + "'", with_error);
        }

        hl::FuncOp declare(mangled_name_ref mangled, auto vast_decl_builder) {
            return declare< hl::FuncOp >(funcdecls, mangled, vast_decl_builder, mangled.name);
        }

        mlir_value declare(const clang::VarDecl *decl, mlir_value vast_value) {
            return declare< mlir_value >(vars, decl, [vast_value] { return vast_value; }, decl->getName());
        }

        mlir_value declare(const clang::VarDecl *decl, auto vast_decl_builder) {
            return declare< mlir_value >(vars, decl, vast_decl_builder, decl->getName());
        }

        hl::LabelDeclOp declare(const clang::LabelDecl *decl, auto vast_decl_builder) {
            return declare< hl::LabelDeclOp >(labels, decl, vast_decl_builder, decl->getName());
        }

        hl::TypeDefOp declare(const clang::TypedefDecl *decl, auto vast_decl_builder) {
            return declare< hl::TypeDefOp >(typedefs, decl, vast_decl_builder, decl->getName());
        }

        hl::TypeDeclOp declare(const clang::TypeDecl *decl, auto vast_decl_builder) {
            return declare< hl::TypeDeclOp >(typedecls, decl, vast_decl_builder, decl->getName());
        }

        hl::EnumDeclOp declare(const clang::EnumDecl *decl, auto vast_decl_builder) {
            return declare< hl::EnumDeclOp >(enumdecls, decl, vast_decl_builder, decl->getName());
        }

        hl::EnumConstantOp declare(const clang::EnumConstantDecl *decl, auto vast_decl_builder) {
            return declare< hl::EnumConstantOp >(enumconsts, decl, vast_decl_builder, decl->getName());
        }

        template< typename SymbolValue, typename Key >
        SymbolValue declare(auto &table, Key &&key, auto vast_decl_builder, string_ref name) {
            if (auto con = table.lookup(key)) {
                return con;
            }

            auto value = vast_decl_builder();
            if (failed(table.declare(std::forward< Key >(key), value))) {
                error("error: multiple declarations with the same name: " + name);
            }

            return value;
        }

        //
        // Integer Attribute Constants
        //
        template< typename T >
        Type bitwidth_type() { return mlir::IntegerType::get(&mctx, bits< T >()); }

        template< typename T >
        mlir::IntegerAttr interger_attr(T v) { return mlir::IntegerAttr::get(bitwidth_type< T >(), v); }

        mlir::IntegerAttr  u8(uint8_t  v) { return interger_attr(v); }
        mlir::IntegerAttr u16(uint16_t v) { return interger_attr(v); }
        mlir::IntegerAttr u32(uint32_t v) { return interger_attr(v); }
        mlir::IntegerAttr u64(uint64_t v) { return interger_attr(v); }

        mlir::IntegerAttr  i8(int8_t  v) { return interger_attr(v); }
        mlir::IntegerAttr i16(int16_t v) { return interger_attr(v); }
        mlir::IntegerAttr i32(int32_t v) { return interger_attr(v); }
        mlir::IntegerAttr i64(int64_t v) { return interger_attr(v); }

        void dump_module() { mod->dump(); }
    };
} // namespace vast::cg
