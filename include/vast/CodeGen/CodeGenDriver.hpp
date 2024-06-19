// Copyright (c) 2021-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <clang/AST/Decl.h>
#include <clang/AST/GlobalDecl.h>
VAST_UNRELAX_WARNINGS

#include "vast/Dialect/HighLevel/HighLevelDialect.hpp"
#include "vast/Dialect/HighLevel/HighLevelOps.hpp"

#include "vast/CodeGen/ScopeContext.hpp"
#include "vast/CodeGen/CodeGenModule.hpp"

#include "vast/Frontend/Options.hpp"

#include "vast/Util/Common.hpp"
#include "vast/Util/DataLayout.hpp"

namespace vast::cg {

    std::unique_ptr< codegen_builder > mk_codegen_builder(mcontext_t &mctx);

    std::shared_ptr< meta_generator > mk_meta_generator(
        acontext_t &actx, mcontext_t &mctx, const cc::vast_args &vargs
    );

    std::shared_ptr< symbol_generator > mk_symbol_generator(
        acontext_t &actx, mcontext_t &mctx, const cc::vast_args &vargs
    );

    std::unique_ptr< mcontext_t > mk_mcontext();

    void set_target_triple(owning_module_ref &mod, std::string triple);
    void set_source_language(owning_module_ref &mod, cc::source_language lang);

    owning_module_ref mk_module(acontext_t &actx, mcontext_t &mctx);
    owning_module_ref mk_module_with_attrs(acontext_t &actx, mcontext_t &mctx, cc::source_language lang);

    struct driver
    {
        explicit driver(
              acontext_t &_actx
            , mcontext_t &_mctx
            , std::unique_ptr< codegen_builder > _bld
            , std::shared_ptr< visitor_base > _visitor
        )
            : actx(_actx)
            , mctx(_mctx)
            , bld(std::move(_bld))
            , visitor(std::move(_visitor))
            , mod(mk_module_with_attrs(
                actx, mctx, cc::get_source_language(actx.getLangOpts())
            ))
            , scope(symbols)
            , generator(*bld, scoped_visitor_view(*visitor, scope))
        {
            bld->module = mod.get();
            bld->set_insertion_point_to_start(&mod->getBodyRegion());
        }

        virtual ~driver() = default;

        bool enable_verifier(bool set = true) { return (enabled_verifier = set); }

        virtual void emit(clang::DeclGroupRef decls);
        virtual void emit(clang::Decl *decl);

        virtual void emit_data_layout();
        virtual void finalize();

        owning_module_ref freeze();

        mcontext_t &mcontext() { return mctx; }
        acontext_t &acontext() { return actx; }

        virtual bool verify();

      private:
        //
        // driver options
        //
        bool enabled_verifier;

        //
        // contexts
        //
        acontext_t &actx;
        mcontext_t &mctx;

        symbol_tables symbols;

        //
        // generators
        //
        std::unique_ptr< codegen_builder > bld;
        std::shared_ptr< visitor_base > visitor;

        //
        // module generation state
        //
        owning_module_ref mod;
        module_scope scope;

        module_generator generator;
    };

    std::unique_ptr< driver > mk_default_driver(
        cc::action_options &opts, const cc::vast_args &vargs,
        acontext_t &actx, mcontext_t &mctx);
} // namespace vast::cg
