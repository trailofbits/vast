// Copyright (c) 2023-present, Trail of Bits, Inc.

#include "vast/Frontend/Pipelines.hpp"

#include "vast/Conversion/Passes.hpp"
#include "vast/Conversion/Passes.hpp"

#include <gap/core/overloads.hpp>

namespace vast::cc {

    namespace pipeline {

        // Generates almost AST like MLIR, without any conversions applied
        pipeline_step_ptr high_level() {
            return conv::pipeline::splice_trailing_scopes();
        }

        // Simplifies high level MLIR
        pipeline_step_ptr reduce_high_level() {
            return compose("reduce-hl",
                conv::pipeline::simplify
            );
        }

        // Generates MLIR with standard types
        pipeline_step_ptr standard_types() {
            return compose("standard-types",
                conv::pipeline::stdtypes
            ).depends_on(reduce_high_level);
        }

        pipeline_step_ptr abi() {
            return conv::pipeline::abi();
        }

        // Conversion to LLVM dialects
        pipeline_step_ptr llvm() {
            return conv::pipeline::to_llvm();
        }

        gap::generator< pipeline_step_ptr > codegen() {
            // TODO: pass further options to augment high level MLIR
            co_yield high_level();
        }

        using noguard_t = std::monostate;
        using guard   = std::variant< string_ref, noguard_t >;

        static constexpr noguard_t noguard = {};

        // Defines a sequence of dialects and for each conversion dialect a
        // passes to be run. This allows to stop conversion at any point (i.e.,
        // specify some intermediate taarget dialect) but produce all dependend
        // steps before.
        using conversion_path = std::vector<
            std::tuple<
                target_dialect, /* target dialect */
                guard,          /* guard to check if the step should be run */
                std::vector< llvm::function_ref< pipeline_step_ptr() > > /* pipeline for the step */
             >
        >;

        bool check_step_guard(guard g, const vast_args &vargs) {
            return std::visit( gap::overloaded {
                [] (noguard_t) { return true; },
                [&] (string_ref opt) { return vargs.has_option(opt); }
            }, g);
        }

        conversion_path default_conversion_path = {
            { target_dialect::high_level, opt::simplify, { reduce_high_level } },
            { target_dialect::std , noguard, { standard_types } },
            { target_dialect::abi , noguard, { abi } },
            { target_dialect::llvm, noguard, { llvm } }
        };

        gap::generator< pipeline_step_ptr > conversion(
            pipeline_source src,
            target_dialect trg,
            const vast_args &vargs
        ) {
            // TODO: add support for custom conversion paths
            // deduced from source, target and vargs
            const auto path = default_conversion_path;

            for (const auto &[dialect, guard, step_passes] : path) {
                if (check_step_guard(guard, vargs)) {
                    for (auto &step : step_passes) {
                        co_yield step();
                    }
                }

                if (trg == dialect) {
                    break;
                }
            }

            if (vargs.has_option(opt::canonicalize)) {
                co_yield conv::pipeline::canonicalize();
            }
        }

    } // namespace pipeline

    bool vast_pipeline::is_disabled(const pipeline_step_ptr &step) const {
        auto disable_step_option = opt::disable(step->name()).str();
        return vargs.has_option(disable_step_option);
    }

    void vast_pipeline::schedule(pipeline_step_ptr step) {
        if (is_disabled(step)) {
            VAST_PIPELINE_DEBUG("step is disabled: {0}", step->name());
            return;
        }

        for (auto &&dep : step->dependencies()) {
            schedule(std::move(dep));
        }

        step->schedule_on(*this);
    }

    std::unique_ptr< vast_pipeline > setup_pipeline(
        pipeline_source src,
        target_dialect trg,
        mcontext_t &mctx,
        const vast_args &vargs
    ) {
        auto passes = std::make_unique< vast_pipeline >(mctx, vargs);

        passes->enableIRPrinting(
            [](auto *, auto *) { return false; }, // before
            [](auto *, auto *) { return true; },  // after
            false,                                // module scope
            false,                                // after change
            true,                                 // after failure
            llvm::errs()
        );

        // generate high level MLIR in case of AST input
        if (pipeline_source::ast == src) {
            for (auto &&step : pipeline::codegen()) {
                passes->schedule(std::move(step));
            }
        }

        // Apply desired conversion to target dialect, if target is llvm or
        // binary/assembly. We perform entire conversion to llvm dialect. Vargs
        // can specify how we want to convert to llvm dialect and allows to turn
        // off optional pipelines.
        for (auto &&step : pipeline::conversion(src, trg, vargs)) {
            passes->schedule(std::move(step));
        }

        if (vargs.has_option(opt::print_pipeline)) {
            passes->dump();
        }

        if (vargs.has_option(opt::disable_multithreading) || vargs.has_option(opt::emit_crash_reproducer)) {
            mctx.disableMultithreading();
        }

        if (vargs.has_option(opt::emit_crash_reproducer)) {
            auto reproducer_path = vargs.get_option(opt::emit_crash_reproducer);
            VAST_CHECK(reproducer_path.has_value(), "expected path to reproducer");
            passes->enableCrashReproducerGeneration(reproducer_path.value(), true /* local reproducer */);
        }

        return passes;
    }

} // namespace vast::cc
