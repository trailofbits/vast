
// Copyright (c) 2023, Trail of Bits, Inc.

#include "vast/Util/Pipeline.hpp"

#include "vast/Dialect/Core/CoreOps.hpp"

namespace vast {

    void pipeline_t::addPass(owning_pass_ptr pass) {
        auto id = pass->getTypeID();
        if (seen.count(id)) {
            return;
        }

        seen.insert(id);
        VAST_PIPELINE_DEBUG("scheduling pass: {0}", pass->getArgument());

        base::addPass(std::move(pass));
    }

    gap::generator< pipeline_step_ptr > pipeline_step::dependencies() const {
        for (const auto &dep : deps) {
            co_yield dep();
        }
    }

    pass_ptr cached_pass_builder::get() const {
        if (!cached_pass) {
            cached_pass = bld();
        }

        return cached_pass.get();
    }

    owning_pass_ptr cached_pass_builder::take() {
        if (cached_pass) {
            return std::move(cached_pass);
        } else {
            return bld();
        }
    }

    gap::generator< pipeline_step_ptr > pass_pipeline_step::substeps() const { co_return; }

    gap::generator< pipeline_step_ptr > compound_pipeline_step::substeps() const {
        for (const auto &step : steps) {
            co_yield step();
        }
    }

    schedule_result pass_pipeline_step::schedule_on(pipeline_t &ppl) {
        ppl.addNestedPass< core::module >(take_pass());
        return schedule_result::advance;
    }

    string_ref pass_pipeline_step::name() const {
        return pass()->getName();
    }

    string_ref pass_pipeline_step::cli_name() const {
        return pass()->getArgument();
    }

    schedule_result top_level_pass_pipeline_step::schedule_on(pipeline_t &ppl) {
        ppl.addPass(take_pass());
        return schedule_result::advance;
    }


    schedule_result compound_pipeline_step::schedule_on(pipeline_t &ppl) {
        VAST_PIPELINE_DEBUG("scheduling compound step: {0}", pipeline_name);
        for (const auto &step : steps) {
            if (ppl.schedule(step()) == schedule_result::stop) {
                return schedule_result::stop;
            }
        }

        return schedule_result::advance;
    }

    string_ref compound_pipeline_step::name() const {
        return pipeline_name;
    }

    string_ref compound_pipeline_step::cli_name() const {
        return pipeline_name;
    }

} // namespace vast
