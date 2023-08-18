// Copyright (c) 2023-present, Trail of Bits, Inc.

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
VAST_UNRELAX_WARNINGS

#include "vast/Dialect/HighLevel/Passes.hpp"
#include "vast/CodeGen/ABIInfo.hpp"
#include "vast/CodeGen/Passes.hpp"

namespace vast::cg {

    logical_result emit_high_level_pass(
        vast_module mod, mcontext_t *mctx, acontext_t */* actx */, bool enable_verifier
    ) {
        mlir::PassManager mgr(mctx);

        // TODO: setup vast intermediate codegen passes
        mgr.addPass(hl::createSpliceTrailingScopes());

        mgr.enableVerifier(enable_verifier);
        return mgr.run(mod);
    }

} // namespace vast::cg
