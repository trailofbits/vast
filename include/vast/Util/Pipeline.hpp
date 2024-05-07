// Copyright (c) 2023-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <llvm/ADT/DenseSet.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/Pass/PassInstrumentation.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Pass/PassRegistry.h>
#include <mlir/Pass/Pass.h>
VAST_UNRELAX_WARNINGS

#include "vast/Util/Common.hpp"

#include <gap/coro/generator.hpp>

namespace vast {

#if !defined(NDEBUG)
    constexpr bool debug_pipelines = false;
    #define VAST_PIPELINE_DEBUG(...) VAST_REPORT_WITH_PREFIX_IF(debug_pipelines, "[pipeline] ", __VA_ARGS__)
#else
    #define VAST_PIPELINE_DEBUG(...)
#endif


    //
    // pipeline_step is a single step in the pipeline that is a pass or a list
    // of pipelines
    //
    // Each step defiens a list of dependencies, which are scheduled before the step
    //
    struct pipeline_step;

    using pipeline_step_ptr = std::unique_ptr< pipeline_step >;

    //
    // pipeline is a pass manager, which keeps track of duplicit passes and does
    // not schedule them twice
    //
    struct pipeline_t : mlir::PassManager
    {
        using base      = mlir::PassManager;
        using pass_id_t = mlir::TypeID;

        using base::base;

        virtual ~pipeline_t() = default;

        void addPass(std::unique_ptr< mlir::Pass > pass);

        template< typename parent_t >
        void addNestedPass(std::unique_ptr< mlir::Pass > pass) {
            auto id = pass->getTypeID();
            if (seen.count(id)) {
                return;
            }

            seen.insert(id);
            VAST_PIPELINE_DEBUG("scheduling nested pass: {0}", pass->getName());
            base::addNestedPass< parent_t >(std::move(pass));
        }

        virtual void schedule(pipeline_step_ptr step) = 0;

        void print_on_error(llvm::raw_ostream &os) {
            enableIRPrinting(
                [](auto *, auto *) { return false; }, // before
                [](auto *, auto *) { return true; },  // after
                false,                                // module scope
                false,                                // after change
                true,                                 // after failure
                os
            );
        }

        llvm::DenseSet< pass_id_t > seen;
    };

    struct with_snapshots : mlir::PassInstrumentation
    {
        using output_stream_ptr = std::shared_ptr< llvm::raw_pwrite_stream >;
        using passes_t = std::vector< llvm::StringRef >;

        std::string file_prefix;

        with_snapshots(llvm::StringRef file_prefix)
            : file_prefix(file_prefix.str())
        {}

        // We return `shared_ptr` in case we may want to keep the stream open for longer
        // in some derived class. Should not make a difference, snapshoting will be expensive
        // anyway.
        virtual output_stream_ptr make_output_stream(mlir::Pass *pass) {
            std::string name = file_prefix + "." + pass->getArgument().str();
            std::error_code ec;
            auto os = std::make_shared< llvm::raw_fd_ostream >(name, ec);
            VAST_CHECK(!ec, "Cannot open file to store snapshot at, ec: {0}", ec.message());
            return std::move(os);
        }

        virtual bool should_snapshot(mlir::Pass *pass) const = 0;

        void runAfterPass(mlir::Pass *pass, operation op) override {
            if (!should_snapshot(pass))
                return;

            auto os = make_output_stream(pass);
            (*os) << *op;
        }
    };

    struct snapshot_at_passes : with_snapshots {
        using base = with_snapshots;

        passes_t snapshot_at;

        template< typename ... Args >
        snapshot_at_passes(const passes_t &snapshot_at, Args && ... args)
            : base(std::forward< Args >(args)...), snapshot_at(snapshot_at)
        {}

        virtual bool should_snapshot(mlir::Pass *pass) const override {
            return std::ranges::count(snapshot_at, pass->getArgument());
        }
    };

    struct snapshot_all : with_snapshots {
        using base = with_snapshots;
        using base::base;

        bool should_snapshot(mlir::Pass *) const override {
            return true;
        }
    };


    using pipeline_step_builder = std::function< pipeline_step_ptr(void) >;

    //
    // initilizer wrapper to setup dependencies after make is called
    //
    template< typename step_t >
    struct pipeline_step_init : pipeline_step_ptr {
        using pipeline_step_ptr::pipeline_step_ptr;

        template< typename ...args_t >
        pipeline_step_init(args_t &&...args)
            : pipeline_step_ptr(std::make_unique< step_t >(
                std::forward< args_t >(args)...
            ))
        {}

        template< typename ...deps_t >
        pipeline_step_init depends_on(deps_t &&... deps) && {
            get()->depends_on(std::forward< deps_t >(deps)...);
            return std::move(*this);
        }
    };

    struct pipeline_step
    {
        explicit pipeline_step(
            std::vector< pipeline_step_builder > deps
        )
            : deps(std::move(deps))
        {}

        explicit pipeline_step() = default;
        virtual ~pipeline_step() = default;

        virtual void schedule_on(pipeline_t &ppl) const = 0;
        virtual gap::generator< pipeline_step_ptr > substeps() const = 0;

        gap::generator< pipeline_step_ptr > dependencies() const;

        virtual string_ref name() const = 0;

        template< typename ...deps_t >
        void depends_on(deps_t &&... dep) {
            (deps.emplace_back(std::forward< deps_t >(dep)), ...);
        }

        std::vector< pipeline_step_builder > deps;
    };

    using pass_builder_t = llvm::function_ref< std::unique_ptr< mlir::Pass >(void) >;

    struct pass_pipeline_step : pipeline_step
    {
        explicit pass_pipeline_step(pass_builder_t builder)
            : pass_builder(builder)
        {}

        void schedule_on(pipeline_t &ppl) const override;

        gap::generator< pipeline_step_ptr > substeps() const override;

        string_ref name() const override;

    protected:
        pass_builder_t pass_builder;
    };

    template< typename parent_t >
    struct nested_pass_pipeline_step : pass_pipeline_step
    {
        explicit nested_pass_pipeline_step(pass_builder_t builder)
            : pass_pipeline_step(builder)
        {}

        virtual ~nested_pass_pipeline_step() = default;

        void schedule_on(pipeline_t &ppl) const override {
            ppl.addNestedPass< parent_t >(pass_builder());
        }
    };

    // compound step represents subpipeline to be run
    struct compound_pipeline_step : pipeline_step
    {
        template< typename... steps_t >
        explicit compound_pipeline_step(string_ref name, steps_t &&...steps)
            : pipeline_name(name), steps{ std::forward< steps_t >(steps)... }
        {}

        virtual ~compound_pipeline_step() = default;

        void schedule_on(pipeline_t &ppl) const override;

        gap::generator< pipeline_step_ptr > substeps() const override;

        string_ref name() const override;

    protected:
        std::string pipeline_name;
        std::vector< pipeline_step_builder > steps;
    };

    template< typename... args_t >
    decltype(auto) pass(args_t &&... args) {
        return pipeline_step_init< pass_pipeline_step >(
            std::forward< args_t >(args)...
        );
    }

    template< typename parent, typename... args_t >
    decltype(auto) nested(args_t &&... args) {
        return pipeline_step_init< nested_pass_pipeline_step< parent > >(
            std::forward< args_t >(args)...
        );
    }

    template< typename... steps_t >
    decltype(auto) compose(string_ref name, steps_t &&...steps) {
        return pipeline_step_init< compound_pipeline_step >(
            name, std::forward< steps_t >(steps)...
        );
    }

} // namespace vast
