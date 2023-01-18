// Copyright (c) 2023-present, Trail of Bits, Inc.

#include "vast/Util/Warnings.hpp"

#include "vast/CodeGen/Passes.hpp"
#include "vast/Frontend/Common.hpp"

namespace vast::cg {

    logical_result emit_high_level_pass(
        vast_module /* mod */, mcontext_t */* mctx */, acontext_t */* actx */, bool /* enable_verifier */
    ) {
        throw cc::compiler_error("emit_high_level_pass not implemented");
    }

} // namespace vast::cg
