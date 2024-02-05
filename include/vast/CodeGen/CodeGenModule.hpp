// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <clang/AST/GlobalDecl.h>
#include <clang/Basic/TargetInfo.h>
VAST_UNRELAX_WARNINGS

#include "vast/Util/Common.hpp"
#include "vast/Util/Triple.hpp"
#include "vast/CodeGen/ScopeContext.hpp"
#include "vast/CodeGen/VisitorView.hpp"
#include "vast/CodeGen/CodeGenMeta.hpp"

#include "vast/Dialect/Core/CoreAttributes.hpp"

namespace vast::cg {

    using source_language = core::SourceLanguage;

    struct module_context : module_scope {
        explicit module_context(owning_module_ref mod)
            : mod(std::move(mod))
        {}

        void set_triple(std::string triple);
        void set_source_language(source_language lang);

        owning_module_ref freeze();

    protected:
        bool frozen = false;
        owning_module_ref mod;
    };

    struct module_generator : module_context
    {
        explicit module_generator(acontext_t &actx, mcontext_t &mctx, source_language lang, meta_generator &meta)
            : module_context(mk_module(actx, mctx)), meta(meta)
        {
            // TODO: set_source_language(lang);
            set_triple(actx.getTargetInfo().getTriple().str());
        }

        void emit(clang::DeclGroupRef decls);
        void emit(clang::Decl *decl);
        void emit(clang::GlobalDecl *decl);
        void emit(clang::TypedefDecl *decl);
        void emit(clang::EnumDecl *decl);
        void emit(clang::RecordDecl *decl);
        void emit(clang::FunctionDecl *decl);
        void emit(clang::VarDecl *decl);

        bool verify();

        void finalize();
    private:
        static owning_module_ref mk_module(acontext_t &actx, mcontext_t &mctx);

        meta_generator &meta;
    };

} // namespace vast::cg
