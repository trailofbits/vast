
// Copyright (c) 2023, Trail of Bits, Inc.

#include "vast/Util/Pipeline.hpp"

namespace vast {

    void pipeline_t::addPass(std::unique_ptr<mlir::Pass> pass) {
        auto id = pass->getTypeID();
        if (seen.count(id)) {
            return;
        }

        seen.insert(id);
        base::addPass(std::move(pass));
    }

    pipeline_t &operator<<(pipeline_t &ppl, pipeline_step_ptr pass) {
        pass->schedule_on(ppl);
        return ppl;
    }

    void pipeline_step::schedule_on(pipeline_t &) const {}

    void pipeline_step::schedule_dependencies(pipeline_t &ppl) const {
        for (const auto &dep : dependencies) {
            dep()->schedule_on(ppl);
        }
    }

    void pass_pipeline_step::schedule_on(pipeline_t &ppl) const {
        schedule_dependencies(ppl);
        ppl.addPass(pass_builder());
    }

    void compound_pipeline_step::schedule_on(pipeline_t &ppl) const {
        schedule_dependencies(ppl);
        for (const auto &step : steps) {
            step()->schedule_on(ppl);
        }
    }

    void optional_pipeline::schedule_on(pipeline_t &ppl) const {
        if (enabled) {
            schedule_dependencies(ppl);
            step()->schedule_on(ppl);
        }
    }

} // namespace vast
