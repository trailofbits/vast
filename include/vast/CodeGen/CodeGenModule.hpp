// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <clang/AST/GlobalDecl.h>
#include <clang/Basic/TargetInfo.h>
#include <mlir/InitAllDialects.h>
VAST_UNRELAX_WARNINGS

#include "vast/Util/Common.hpp"
#include "vast/Util/Triple.hpp"
#include "vast/Util/DataLayout.hpp"

#include "vast/CodeGen/ScopeContext.hpp"
#include "vast/CodeGen/VisitorView.hpp"
#include "vast/CodeGen/CodeGenMeta.hpp"

#include "vast/Dialect/Dialects.hpp"
#include "vast/Dialect/Core/CoreAttributes.hpp"

#include "vast/CodeGen/Mangler.hpp"

namespace vast::cg {

    using source_language = core::SourceLanguage;

    void set_target_triple(owning_module_ref &mod, std::string triple);
    void set_source_language(owning_module_ref &mod, source_language lang);

    struct module_context : module_scope {
        explicit module_context(owning_module_ref mod, acontext_t &actx, scope_context *parent = nullptr)
            : module_scope(parent)
            , mod(std::move(mod))
            , mangler(actx.createMangleContext())
        {}

        virtual ~module_context() = default;

        owning_module_ref freeze();

        owning_module_ref mod;
        dl::DataLayoutBlueprint dl;
        mangler_t mangler;
    };

    owning_module_ref mk_module(acontext_t &actx, mcontext_t &mctx);
    owning_module_ref mk_module_with_attrs(acontext_t &actx, mcontext_t &mctx, source_language lang);

    const target_info &get_target_info(const module_context *mod);
    const std::string &get_module_name_hash(const module_context *mod);
    mangled_name_ref get_mangled_name(const module_context *mod, clang_function *decl);


    struct module_generator : module_context
    {
        explicit module_generator(acontext_t &actx, mcontext_t &mctx, source_language lang, meta_generator &meta)
            : module_context(mk_module_with_attrs(actx, mctx, lang), actx), meta(meta)
        {}

        virtual ~module_generator() = default;

        void emit(clang::DeclGroupRef decls);
        void emit(clang::Decl *decl);
        void emit(clang::GlobalDecl *decl);
        void emit(clang::TypedefDecl *decl);
        void emit(clang::EnumDecl *decl);
        void emit(clang::RecordDecl *decl);
        void emit(clang::FunctionDecl *decl);
        void emit(clang::VarDecl *decl);

        void emit_data_layout();

        bool verify();
        void finalize();
    private:
        [[maybe_unused]] meta_generator &meta;
    };

} // namespace vast::cg
