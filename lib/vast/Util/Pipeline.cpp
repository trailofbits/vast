
// Copyright (c) 2023, Trail of Bits, Inc.

#include "vast/Util/Pipeline.hpp"

namespace vast {

    void pipeline_t::addPass(std::unique_ptr<mlir::Pass> pass) {
        auto id = pass->getTypeID();
        if (seen.count(id)) {
            return;
        }

        seen.insert(id);
        VAST_PIPELINE_DEBUG("scheduling pass: {0}", pass->getName());
        base::addPass(std::move(pass));
    }

    gap::generator< pipeline_step_ptr > pipeline_step::dependencies() const {
        for (const auto &dep : deps) {
            co_yield dep();
        }
    }

    gap::generator< pipeline_step_ptr > pass_pipeline_step::substeps() const { co_return; }

    gap::generator< pipeline_step_ptr > compound_pipeline_step::substeps() const {
        for (const auto &step : steps) {
            co_yield step();
        }
    }

    void pass_pipeline_step::schedule_on(pipeline_t &ppl) const {
        ppl.addPass(pass_builder());
    }

    string_ref pass_pipeline_step::name() const {
        return pass_builder()->getName();
    }
    string_ref pass_pipeline_step::cli_name() const {
        return pass_builder()->getArgument();
    }

    void compound_pipeline_step::schedule_on(pipeline_t &ppl) const {
        VAST_PIPELINE_DEBUG("scheduling compound step: {0}", pipeline_name);
        for (const auto &step : steps) {
            ppl.schedule(step());
        }
    }

    string_ref compound_pipeline_step::name() const {
        return pipeline_name;
    }

    string_ref compound_pipeline_step::cli_name() const {
        return pipeline_name;
    }

} // namespace vast
