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
#include "vast/CodeGen/ScopeGenerator.hpp"
#include "vast/CodeGen/CodeGenMeta.hpp"
#include "vast/CodeGen/CodeGenOptions.hpp"

#include "vast/CodeGen/CodeGenFunction.hpp"

#include "vast/Dialect/Dialects.hpp"
#include "vast/Dialect/Core/CoreAttributes.hpp"

#include "vast/CodeGen/Mangler.hpp"

namespace vast::cg {

    void set_target_triple(owning_module_ref &mod, std::string triple);
    void set_source_language(owning_module_ref &mod, source_language lang);

    owning_module_ref mk_module(acontext_t &actx, mcontext_t &mctx);
    owning_module_ref mk_module_with_attrs(acontext_t &actx, mcontext_t &mctx, source_language lang);

    struct module_context : module_scope {
        explicit module_context(
              symbol_tables &scopes
            , const options_t &opts
            , acontext_t &actx
            , mcontext_t &mctx
        )
            : module_scope(scopes)
            , opts(opts)
            , actx(actx)
            , mod(mk_module_with_attrs(actx, mctx, opts.lang))
        {}

        virtual ~module_context() = default;

        owning_module_ref freeze();

        operation lookup_global(symbol_name name) const;

        const options_t &opts;

        acontext_t &actx;
        owning_module_ref mod;
    };

    struct module_generator : scope_generator< module_generator, module_context >
    {
        using base = scope_generator< module_generator, module_context >;

        explicit module_generator(
              acontext_t &actx
            , mcontext_t &mctx
            , const options_t &opts
            , codegen_builder &bld
            , visitor_view visitor
            , symbol_tables &scopes
        )
            : base(visitor, bld, scopes, opts, actx, mctx)
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
    };

} // namespace vast::cg
