// Copyright (c) 2023-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <llvm/ADT/DenseSet.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Pass/PassRegistry.h>
#include <mlir/Pass/Pass.h>
VAST_UNRELAX_WARNINGS

#include "vast/Util/Common.hpp"

#include "gap/core/generator.hpp"

namespace vast {

    //
    // pipeline is a pass manager, which keeps track of duplicit passes and does
    // not schedule them twice
    //
    struct pipeline_t : mlir::PassManager
    {
        using base      = mlir::PassManager;
        using pass_id_t = mlir::TypeID;

        void addPass(std::unique_ptr< mlir::Pass > pass) {
            auto id = pass->getTypeID();
            if (seen.count(id)) {
                return;
            }

            seen.insert(id);
            base::addPass(std::move(pass));
        }

        llvm::DenseSet< pass_id_t > seen;
    };


    //
    // pipeline_step is a single step in the pipeline that is a pass or a list
    // of pipelines
    //
    // Each step defiens a list of dependencies, which are scheduled before the step
    //
    struct pipeline_step;

    using pipeline_step_ptr = std::unique_ptr< pipeline_step >;

    using pipeline_step_builder = llvm::function_ref< pipeline_step_ptr(void) >;

    //
    // initilizer wrapper to setup dependencies after make is called
    //
    template< typename step_t >
    struct pipeline_step_init : pipeline_step_ptr {
        using pipeline_step_ptr::pipeline_step_ptr;

        template< typename ...Args >
        pipeline_step_init(Args &&...args)
            : pipeline_step_ptr(std::make_unique< step_t >(
                std::forward< Args >(args)...
            ))
        {}

        pipeline_step_init depends_on(std::vector< pipeline_step_builder > deps) && {
            get()->depends_on(std::move(deps));
            return std::move(*this);
        }
    };

    struct pipeline_step
    {
        explicit pipeline_step(
            std::vector< pipeline_step_builder > dependencies
        )
            : dependencies(std::move(dependencies))
        {}

        explicit pipeline_step() = default;

        virtual ~pipeline_step() = default;

        virtual void schedule_on(pipeline_t &ppl) const = 0;

        void schedule_dependencies(pipeline_t &ppl) const;

        void depends_on(std::vector< pipeline_step_builder > deps);

        std::vector< pipeline_step_builder > dependencies;
    };

    using pass_builder_t = llvm::function_ref< std::unique_ptr< mlir::Pass >(void) >;

    struct pass_pipeline_step : pipeline_step
    {
        explicit pass_pipeline_step(pass_builder_t builder)
            : pass_builder(builder)
        {}

        void schedule_on(pipeline_t &ppl) const override;

        static decltype(auto) make(pass_builder_t builder) {
            return pipeline_step_init< pass_pipeline_step >(builder);
        }

        pass_builder_t pass_builder;
    };

    // compund step represents subpipeline to be run
    struct compound_pipeline_step : pipeline_step
    {
        explicit compound_pipeline_step(
            std::vector< pipeline_step_builder > steps
        )
            : steps(std::move(steps))
        {}

        void schedule_on(pipeline_t &ppl) const override;

        static decltype(auto) make(std::vector< pipeline_step_builder > dependencies) {
            return pipeline_step_init< compound_pipeline_step >(std::move(dependencies));
        }

        std::vector< pipeline_step_builder > steps;
    };

} // namespace vast
