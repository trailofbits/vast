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
        explicit module_context(owning_module_ref mod, acontext_t &actx)
            : mod(std::move(mod))
            , mangler(actx.createMangleContext())
        {}

        owning_module_ref freeze();

    protected:
        bool frozen = false;
        owning_module_ref mod;
        dl::DataLayoutBlueprint dl;

        mangler_t mangler;
    };

    owning_module_ref mk_module(acontext_t &actx, mcontext_t &mctx);
    owning_module_ref mk_module_with_attrs(acontext_t &actx, mcontext_t &mctx, source_language lang);


    struct module_generator : module_context
    {
        explicit module_generator(acontext_t &actx, mcontext_t &mctx, source_language lang, meta_generator &meta)
            : module_context(mk_module_with_attrs(actx, mctx, lang), actx), meta(meta)
        {}

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
