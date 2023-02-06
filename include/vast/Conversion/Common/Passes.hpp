// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include "mlir/Transforms/DialectConversion.h"
VAST_UNRELAX_WARNINGS

#include "vast/Util/LLVMTypeConverter.hpp"

#include <vast/Conversion/Common/Passes.hpp>

namespace vast {

    using conversion_target = mlir::ConversionTarget;

    //
    // Mixin to define simple module conversion passes. It requires the derived
    // pass to define two static methods.
    //
    // To specify the legalization and illegalization of operations:
    //
    // `static conversion_target create_conversion_target(MContext &context)`
    //
    // To populate rewrite patterns:
    //
    // `static void populate_conversions(rewrite_pattern_set &patterns)`
    //
    // The mixin provides the derived pass by populate_conversions helper, which
    // takes lists of rewrite patterns.
    //
    // Example usage:
    //
    // struct ExamplePass : ModuleConversionPassMixin< ExamplePass, ExamplePassBase > {
    //     using base = ModuleConversionPassMixin< ExamplePass, ExamplePassBase >;
    //
    //     static conversion_target create_conversion_target(MContext &context) {
    //         conversion_target target(context);
    //         // setup target here
    //         return target;
    //     }
    //
    //     static void populate_conversions(rewrite_pattern_set &patterns) {
    //         base::populate_conversions_base<
    //             // pass conversion type_lists here
    //         >(patterns);
    //     }
    // }
    //
    template< typename derived_t, template< typename > typename base_t >
    struct ModuleConversionPassMixin : base_t< derived_t > {

        using base = base_t< derived_t >;

        using base::getContext;
        using base::getOperation;
        using base::signalPassFailure;

        using rewrite_pattern_set = mlir::RewritePatternSet;

        template< typename list >
        static void populate_conversions_impl(rewrite_pattern_set &patterns) {
            if constexpr ( list::empty ) {
                return;
            } else {
                patterns.add< typename list::head >(patterns.getContext());
                return populate_conversions_base< typename list::tail >(patterns);
            }
        }

        template< typename ...lists  >
        static void populate_conversions_base(rewrite_pattern_set &patterns) {
            (populate_conversions_impl< lists >(patterns), ...);
        }

        void run_on_operation() {
            auto &ctx   = getContext();
            auto target = derived_t::create_conversion_target(ctx);

            // populate all patterns
            rewrite_pattern_set patterns(&ctx);
            derived_t::populate_conversions(patterns);

            // run on operation
            if (failed(applyPartialConversion(getOperation(), target, std::move(patterns))))
                signalPassFailure();
        }

        // interface with pass base
        void runOnOperation() override { run_on_operation(); }
    };

    template< typename derived_t, template< typename > typename base_t >
    struct ModuleLLVMConversionPassMixin : base_t< derived_t > {

        using base = base_t< derived_t >;

        using base::getContext;
        using base::getOperation;
        using base::signalPassFailure;

        using rewrite_pattern_set = mlir::RewritePatternSet;

        using type_converter = util::tc::LLVMTypeConverter;

        template< typename list >
        static void populate_conversions_impl(
            rewrite_pattern_set &patterns, type_converter &converter)
        {
            if constexpr ( list::empty ) {
                return;
            } else {
                patterns.add< typename list::head >(converter);
                return populate_conversions_base< typename list::tail >(patterns, converter);
            }
        }

        template< typename ...lists  >
        static void populate_conversions_base(rewrite_pattern_set &patterns, type_converter &converter) {
            (populate_conversions_impl< lists >(patterns, converter), ...);
        }

        void run_on_operation() {
            auto &ctx   = getContext();
            auto target = derived_t::create_conversion_target(ctx);
            const auto &dl_analysis = this->template getAnalysis< mlir::DataLayoutAnalysis >();

            mlir::LowerToLLVMOptions llvm_options{ &ctx };

            derived_t::set_llvm_opts(llvm_options);

            type_converter converter(&ctx, llvm_options , &dl_analysis);

            // populate all patterns
            rewrite_pattern_set patterns(&ctx);
            derived_t::populate_conversions(patterns, converter);

            // run on operation
            if (failed(applyPartialConversion(getOperation(), target, std::move(patterns))))
                signalPassFailure();
        }

        void runOnOperation() override { run_on_operation(); }
    };
}
