// Copyright (c) 2023-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

#include "vast/Util/Common.hpp"
#include "vast/Frontend/Diagnostics.hpp"

namespace vast::cg {

    struct codegen_module {
        codegen_module(const codegen_module &) = delete;
        codegen_module &operator=(const codegen_module &) = delete;

        codegen_module(
            mcontext_t &/* mctx */, acontext_t &actx,
            cc::diagnostics_engine &diags,
            const cc::codegen_options &cgo
        )
            : actx(actx), diags(diags), lang_opts(actx.getLangOpts()), codegen_opts(cgo)
        {}

        const cc::diagnostics_engine &get_diags() const { return diags; }

        // Finalize vast code generation.
        void release();

      private:
        acontext_t &actx;

        cc::diagnostics_engine &diags;

        const cc::language_options &lang_opts;
        const cc::codegen_options &codegen_opts;
    };

} // namespace vast::cg
