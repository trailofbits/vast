// Copyright (c) 2023-present, Trail of Bits, Inc.

#include "vast/Frontend/Pipelines.hpp"

#include "vast/Dialect/HighLevel/Passes.hpp"
#include "vast/Conversion/Passes.hpp"

namespace vast::cc {

    pipelines_config default_pipelines_config() {
        return pipelines_config{{
            { "canonicalize", hl::pipeline::canonicalize },
            { "desugar", hl::pipeline::desugar },
            { "simplify", hl::pipeline::simplify },
            { "stdtypes", hl::pipeline::stdtypes }
        }};
    }

    namespace pipeline {

        // Generates almost AST like MLIR, without any conversions applied
        pipeline_step_ptr high_level() {
            return hl::pipeline::canonicalize();
        }

        // Simplifies high level MLIR
        pipeline_step_ptr reduce_high_level() {
            return compose(
                hl::pipeline::desugar,
                hl::pipeline::simplify
            );
        }

        // Generates MLIR with standard types
        pipeline_step_ptr standard_types() {
            return hl::pipeline::stdtypes();
        }

        // Conversion to LLVM dialects
        pipeline_step_ptr llvm() {
            return conv::pipeline::to_llvm();
        }

        pipeline_step_ptr codegen() {
            // TODO: pass further options to augment high level MLIR
            return high_level();
        }

        // Defines a sequence of dialects and for each conversion dialect a
        // passes to be run. This allows to stop conversion at any point (i.e.,
        // specify some intermediate taarget dialect) but produce all dependend
        // steps before.
        using conversion_path = std::vector<
            std::pair< target_dialect, std::vector< llvm::function_ref< pipeline_step_ptr() > > >
        >;

        conversion_path default_conversion_path = {
            { target_dialect::high_level, { reduce_high_level } },
            { target_dialect::std, { standard_types } },
            { target_dialect::llvm, { llvm } }
        };

        gap::generator< pipeline_step_ptr > conversion(
            pipeline_source src,
            output_type trg,
            const vast_args &vargs,
            const pipelines_config &config
        ) {
            // TODO: add support for custom conversion paths
            // deduced from source, target and vargs
            auto path = default_conversion_path;

            auto target_dialect =  [&] () -> enum target_dialect {
                if (trg == output_type::emit_mlir) {
                    return parse_target_dialect(vargs.get_option(opt::emit_mlir).value());
                }
                // if we user do not specify target dialect, we convert all the way to llvm
                return target_dialect::llvm;
            } ();

            bool simplify = vargs.has_option(opt::simplify);

            if (target_dialect == target_dialect::high_level && !simplify) {
                co_return;
            }

            for (const auto &[dialect, step_passes] : path) {
                for (auto &step : step_passes) {
                    co_yield step();
                }

                if (target_dialect == dialect) {
                    break;
                }
            }
        }

    } // namespace pipeline

    std::unique_ptr< pipeline_t > setup_pipeline(
        pipeline_source src,
        output_type trg,
        mcontext_t &mctx,
        const vast_args &vargs,
        const pipelines_config &config
    ) {
        auto passes = std::make_unique< pipeline_t >(&mctx);

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
            *passes << pipeline::codegen();
        }

        // Apply desired conversion to target dialect, if target is llvm or
        // binary/assembly. We perform entire conversion to llvm dialect. Vargs
        // can specify how we want to convert to llvm dialect and allows to turn
        // off optional pipelines.
        for (auto &&step : pipeline::conversion(src, trg, vargs, config)) {
            *passes << std::move(step);
        }

        return passes;
    }

} // namespace vast::cc
