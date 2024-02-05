// Copyright (c) 2021-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <clang/AST/Decl.h>
#include <clang/AST/GlobalDecl.h>
VAST_UNRELAX_WARNINGS

#include "vast/Dialect/HighLevel/HighLevelDialect.hpp"
#include "vast/Dialect/HighLevel/HighLevelOps.hpp"

#include "vast/CodeGen/CodeGen.hpp"
#include "vast/CodeGen/CodeGenModule.hpp"

#include "vast/Util/Common.hpp"
#include "vast/Util/DataLayout.hpp"

namespace vast::cg {

    struct driver
    {
        explicit driver(cc::action_options &opts, const cc::vast_args &vargs, acontext_t &actx)
            : actx(actx)
            , mctx(std::make_unique< mcontext_t >())
            , meta(mk_meta_generator(vargs))
            , generator(actx, *mctx, cc::get_source_language(opts.lang), *meta)
        {}

        void emit(clang::DeclGroupRef decls);
        void emit(clang::Decl *decl);
        void finalize(const cc::vast_args &vargs);

        owning_module_ref freeze();

        mcontext_t &mcontext() { return *mctx; }
        acontext_t &acontext() { return actx; }

      private:
        meta_generator_ptr mk_meta_generator(const cc::vast_args &vargs);

        //
        // contexts
        //
        acontext_t &actx;
        std::unique_ptr< mcontext_t > mctx;

        //
        // generators
        //
        meta_generator_ptr meta;
        module_generator generator;
    };

} // namespace vast::cg
