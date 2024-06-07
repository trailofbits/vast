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
#include "vast/CodeGen/CodeGenVisitor.hpp"

#include "vast/Frontend/Options.hpp"

#include "vast/Util/Common.hpp"
#include "vast/Util/DataLayout.hpp"

namespace vast::cg {

    std::unique_ptr< meta_generator > mk_meta_generator(
        acontext_t &actx, mcontext_t &mctx, const cc::vast_args &vargs
    );

    std::unique_ptr< symbol_generator > mk_symbol_generator(
        acontext_t &actx, mcontext_t &mctx, const cc::vast_args &vargs
    );

    std::unique_ptr< mcontext_t > mk_mcontext();

    void set_target_triple(owning_module_ref &mod, std::string triple);
    void set_source_language(owning_module_ref &mod, source_language lang);

    owning_module_ref mk_module(acontext_t &actx, mcontext_t &mctx);
    owning_module_ref mk_module_with_attrs(acontext_t &actx, mcontext_t &mctx, source_language lang);

    struct driver
    {
        explicit driver(
              acontext_t &_actx
            , std::unique_ptr< mcontext_t > _mctx
            , options _opts
            , std::unique_ptr< codegen_builder > _bld
            , std::unique_ptr< meta_generator > _mg
            , std::unique_ptr< symbol_generator > _sg
        )
            : actx(_actx)
            , mctx(std::move(_mctx))
            , opts(std::move(_opts))
            , bld(std::move(_bld))
            , mg(std::move(_mg))
            , sg(std::move(_sg))
            , visitor(mk_visitor(opts))
            , mod(mk_module_with_attrs(actx, *mctx, opts.lang))
            , scope(symbols)
            , generator(*bld, scoped_visitor_view(*visitor, scope), opts)
        {
            bld->module = mod.get();
            bld->set_insertion_point_to_start(&mod->getBodyRegion());
        }

        void emit(clang::DeclGroupRef decls);
        void emit(clang::Decl *decl);
        void finalize(const cc::vast_args &vargs);

        owning_module_ref freeze();

        mcontext_t &mcontext() { return *mctx; }
        acontext_t &acontext() { return actx; }

        bool verify();
      private:

        //
        // contexts
        //
        acontext_t &actx;
        std::unique_ptr< mcontext_t > mctx;

        options opts;
        symbol_tables symbols;

        //
        // generators
        //
        std::unique_ptr< codegen_builder > bld;
        std::unique_ptr< meta_generator > mg;
        std::unique_ptr< symbol_generator > sg;

        std::unique_ptr< codegen_visitor > visitor;
        std::unique_ptr< codegen_visitor > mk_visitor(const options &opts);

        //
        // module generation state
        //
        owning_module_ref mod;
        module_scope scope;

        module_generator generator;
    };

    std::unique_ptr< driver > mk_driver(cc::action_options &opts, const cc::vast_args &vargs, acontext_t &actx);

} // namespace vast::cg
