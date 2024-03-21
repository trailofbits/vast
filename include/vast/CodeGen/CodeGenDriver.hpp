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
#include "vast/CodeGen/CodeGenVisitorBase.hpp"

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

    std::unique_ptr< visitor_base > mk_visitor(
        const cc::vast_args &vargs, mcontext_t &mctx, meta_generator &mg
    );

    struct driver
    {
        explicit driver(
              acontext_t &actx
            , std::unique_ptr< mcontext_t > mctx
            , std::unique_ptr< codegen_builder > bld
            , std::unique_ptr< meta_generator > mg
            , std::unique_ptr< symbol_generator > sg
            , std::unique_ptr< visitor_base > visitor
        )
            : actx(actx)
            , mctx(std::move(mctx))
            , bld(std::move(bld))
            , mg(std::move(mg))
            , sg(std::move(sg))
            , visitor(std::move(visitor))
            , generator(mk_module_generator())
        {}

        void emit(clang::DeclGroupRef decls);
        void emit(clang::Decl *decl);
        void finalize(const cc::vast_args &vargs);

        owning_module_ref freeze();

        mcontext_t &mcontext() { return *mctx; }
        acontext_t &acontext() { return actx; }

      private:

        //
        // contexts
        //
        acontext_t &actx;
        std::unique_ptr< mcontext_t > mctx;

        symbol_tables scopes;

        //
        // generators
        //
        std::unique_ptr< codegen_builder > bld;
        std::unique_ptr< meta_generator > mg;
        std::unique_ptr< symbol_generator > sg;
        std::unique_ptr< visitor_base > visitor;

        //
        // module generation state
        //
        module_generator mk_module_generator();
        module_generator generator;
    };

    std::unique_ptr< driver > mk_driver(cc::action_options &opts, const cc::vast_args &vargs, acontext_t &actx);

} // namespace vast::cg
