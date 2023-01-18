// Copyright (c) 2023-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

#include "vast/Util/Common.hpp"
#include "vast/CodeGen/Builder.hpp"
#include "vast/Frontend/Diagnostics.hpp"

namespace vast::cg {

    struct codegen_module {
        codegen_module(const codegen_module &) = delete;
        codegen_module &operator=(const codegen_module &) = delete;

        codegen_module(
            mcontext_t &mctx, acontext_t &actx,
            cc::diagnostics_engine &diags,
            const cc::codegen_options &cgo
        )
            : builder(mctx), actx(actx)
            , diags(diags), lang_opts(actx.getLangOpts()), codegen_opts(cgo)
            , mod( mlir::ModuleOp::create(builder.getUnknownLoc()) )
        {}

        const cc::diagnostics_engine &get_diags() const { return diags; }

        // Finalize vast code generation.
        void release();

        vast_module get_module() { return mod; }

      private:

        // The builder is a helper class to create IR inside a function. The
        // builder is stateful, in particular it keeps an "insertion point": this
        // is where the next operations will be introduced.
        codegen_builder builder;

        acontext_t &actx;

        cc::diagnostics_engine &diags;

        const cc::language_options &lang_opts;
        const cc::codegen_options &codegen_opts;

        // A "module" matches a c/cpp source file: containing a list of functions.
        vast_module mod;
    };

} // namespace vast::cg
