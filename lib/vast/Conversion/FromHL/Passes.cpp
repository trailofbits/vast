// Copyright (c) 2023-present, Trail of Bits, Inc.

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
VAST_UNRELAX_WARNINGS

#include "vast/Conversion/Passes.hpp"

#include "vast/Dialect/LowLevel/LowLevelOps.hpp"

namespace vast::conv::pipeline {

    pipeline_step_ptr to_hlbi() {
        return pass(createHLToHLBI);
    }

    pipeline_step_ptr hl_to_ll_cf() {
        // TODO add dependencies
        return pass(createHLToLLCFPass);
    }

    pipeline_step_ptr hl_to_ll_geps() {
        // TODO add dependencies
        return pass(createHLToLLGEPsPass);
    }

    pipeline_step_ptr lazy_regions() {
        // TODO add dependencies
        return pass(createHLEmitLazyRegionsPass);
    }

    pipeline_step_ptr hl_to_ll_func() {
        // TODO add dependencies
        return pass(createHLToLLFuncPass);
    }

    // FIXME: move to ToMem/Passes.cpp eventually
    pipeline_step_ptr vars_to_cells() {
        return pass(createVarsToCellsPass);
    }

    pipeline_step_ptr evict_static_locals() {
        return pass(createEvictStaticLocalsPass);
    }

    pipeline_step_ptr refs_to_ssa() {
        return pass(createRefsToSSAPass)
            .depends_on(vars_to_cells);
    }

    // FIXME: run on hl.FuncOp. Once we remove graph regions ll::FuncOp is no longer needed
    pipeline_step_ptr strip_param_lvalues() {
        return pass(createStripParamLValuesPass);
    }

    pipeline_step_ptr to_mem() {
        return compose("to-mem",
            vars_to_cells,
            refs_to_ssa,
            evict_static_locals,
            strip_param_lvalues
        );
    }


    pipeline_step_ptr lower_value_categories() {
        return pass(createLowerValueCategoriesPass)
            .depends_on(to_mem);
    }

    pipeline_step_ptr to_ll() {
        return compose( "to-ll",
            hl_to_ll_func,
            hl_to_ll_cf,
            hl_to_ll_geps,
            lower_value_categories,
            lazy_regions
        );
    }

} // namespace vast::conv::pipeline
