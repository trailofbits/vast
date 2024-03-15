// Copyright (c) 2021-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <clang/AST/Decl.h>
#include <clang/AST/GlobalDecl.h>
VAST_UNRELAX_WARNINGS

#include "vast/Dialect/HighLevel/HighLevelDialect.hpp"
#include "vast/Dialect/HighLevel/HighLevelOps.hpp"

#include "vast/CodeGen/CodeGenModule.hpp"
#include "vast/CodeGen/CodeGenVisitorBase.hpp"

#include "vast/Frontend/Options.hpp"

#include "vast/Util/Common.hpp"
#include "vast/Util/DataLayout.hpp"

namespace vast::cg {

    std::unique_ptr< meta_generator > mk_meta_generator(
        acontext_t &actx, mcontext_t &mctx, const cc::vast_args &vargs
    );

    std::unique_ptr< mcontext_t > mk_mcontext();

    std::unique_ptr< codegen_visitor_base > mk_visitor(const cc::vast_args &vargs);

    struct driver
    {
        explicit driver(cc::action_options &opts, const cc::vast_args &vargs, acontext_t &actx)
            : actx(actx)
            , mctx(mk_mcontext())
            , meta(mk_meta_generator(actx, *mctx, vargs))
            , visitor(mk_visitor(vargs))
            , generator(actx, *mctx, cc::get_source_language(opts.lang), *meta, visitor_view(*visitor))
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

        //
        // generators
        //
        std::unique_ptr< meta_generator > meta;
        std::unique_ptr< codegen_visitor_base > visitor;

        //
        // module generation state
        //
        module_generator generator;
    };

} // namespace vast::cg
