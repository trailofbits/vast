// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include "mlir/Transforms/DialectConversion.h"
VAST_UNRELAX_WARNINGS

#include "vast/Conversion/Common/Patterns.hpp"
#include "vast/Conversion/Common/Types.hpp"
#include "vast/Conversion/TypeConverters/LLVMTypeConverter.hpp"

namespace vast {

    // Inject basic api shared by other mixins:
    //  - iterating over lists of patterns.
    //  - applying the conversion.
    //  Since we cannot easily do `using config_t = self_t::config_t` the type is instead
    //  taken as a template.
    template< typename self_t >
    struct populate_patterns
    {
        template< typename list, typename config_t >
        static void populate_conversions_impl(config_t &config) {
            if constexpr (list::empty) {
                return;
            } else {
                self_t::template add_pattern< typename list::head >(config);
                self_t::template legalize< typename list::head >(config);
                return self_t::template populate_conversions_impl< typename list::tail >(config);
            }
        }

        template< typename pattern, typename config_t >
        static void legalize(config_t &config) {
            if constexpr (has_legalize< pattern >) {
                pattern::legalize(config.target);
            }
        }

        auto &self() { return static_cast< self_t & >(*this); }

        // It is expected to move into this method, as it consumes & runs a configuration.
        template< typename config_t >
        auto apply_conversions(config_t config) {
            return mlir::applyPartialConversion(
                self().getOperation(), config.target, std::move(config.patterns)
            );
        }

        template< typename... lists, typename config_t >
        static void populate_conversions_base(config_t &config) {
            (self_t::template populate_conversions_impl< lists >(config), ...);
        }
    };

    //
    // Mixin to define simple module conversion passes. It requires the derived
    // pass to define two static methods.
    //
    // To specify the legalization and illegalization of operations:
    //
    // `static conversion_target create_conversion_target(mcontext_t &context)`
    //
    // To populate rewrite patterns:
    //
    // `static void populate_conversions(config_t &config)`
    //
    // The mixin provides to the derived pass `populate_conversions` helper, which
    // takes lists of rewrite patterns.
    // Aside from populating collection of patterns, this method also calls `legalize` method
    // of every pattern being added.
    //
    // Example usage:
    //
    // struct ExamplePass : ModuleConversionPassMixin< ExamplePass, ExamplePassBase > {
    //     using base = ModuleConversionPassMixin< ExamplePass, ExamplePassBase >;
    //
    //     static conversion_target create_conversion_target(mcontext_t &context) {
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
    struct ModuleConversionPassMixin
        : base_t< derived_t >
        , populate_patterns< derived_t >
    {
        using base     = base_t< derived_t >;
        using populate = populate_patterns< derived_t >;

        using base::getContext;
        using base::getOperation;
        using base::signalPassFailure;

        using rewrite_pattern_set = mlir::RewritePatternSet;

        struct config_t
        {
            rewrite_pattern_set patterns;
            conversion_target target;

            mlir::MLIRContext *getContext() { return patterns.getContext(); }
        };

        template< typename pattern >
        static void add_pattern(config_t &config) {
            config.patterns.template add< pattern >(config.getContext());
        }

        auto &self() { return static_cast< derived_t & >(*this); }

        // Override
        void populate_conversions(config_t &) {}

        void run_on_operation() {
            auto &ctx = getContext();
            auto config = config_t{
                rewrite_pattern_set(&ctx), derived_t::create_conversion_target(ctx)
            };

            self().populate_conversions(config);

            if (failed(populate::apply_conversions(std::move(config)))) {
                return signalPassFailure();
            }

            this->after_operation();
        }

        // interface with pass base
        void runOnOperation() override { run_on_operation(); }

        // Override to specify what is supposed to run after `run_on_operation` is finished.
        // This will run *only if the `run_on_operation* was successful.
        virtual void after_operation() {};
    };

    // Sibling of the above module for passes that go to the LLVM dialect.
    // Example usage:
    //
    // struct ExamplePass : ModuleLLVMConversionPassMixin< ExamplePass, ExamplePassBase > {
    //     using base = ModuleLLVMConversionPassMixin< ExamplePass, ExamplePassBase >;
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
    //     static void set_llvm_options(mlir::LowerToLLVMOptions &llvm_options) {
    //          llvm_options.myOption = my_value;
    //     }
    // }
    //
    template< typename derived_t, template< typename > typename base_t >
    struct ModuleLLVMConversionPassMixin
        : base_t< derived_t >
        , populate_patterns< derived_t >
    {
        using base     = base_t< derived_t >;
        using populate = populate_patterns< derived_t >;

        using base::getContext;
        using base::getOperation;
        using base::signalPassFailure;

        using rewrite_pattern_set = mlir::RewritePatternSet;

        using llvm_type_converter = conv::tc::FullLLVMTypeConverter;

        struct config
        {
            rewrite_pattern_set patterns;
            conversion_target target;
            // Type converter cannot be moved!
            llvm_type_converter &tc;

            mcontext_t *getContext() { return patterns.getContext(); }

            config(
                rewrite_pattern_set patterns, conversion_target target, llvm_type_converter &tc
            )
                : patterns(std::move(patterns)), target(std::move(target)), tc(tc)
            {}

            config(config &&other)
                : patterns(std::move(other.patterns))
                , target(std::move(other.target))
                , tc(other.tc)
            {}
        };

        auto &self() { return static_cast< derived_t & >(*this); }

        // Override
        void populate_conversions(config &) {}

        template< typename pattern >
        static void add_pattern(config &cfg) {
            cfg.patterns.template add< pattern >(cfg.tc);
        }

        void run_on_operation() {
            auto &ctx               = getContext();
            const auto &dl_analysis = this->template getAnalysis< mlir::DataLayoutAnalysis >();

            mlir::LowerToLLVMOptions llvm_options{ &ctx };
            derived_t::set_llvm_opts(llvm_options);

            auto tc  = llvm_type_converter(getOperation(), &ctx, llvm_options, &dl_analysis);
            auto cfg = config(
                rewrite_pattern_set(&ctx), derived_t::create_conversion_target(ctx, tc), tc
            );

            // populate all patterns
            self().populate_conversions(cfg);

            if (failed(populate::apply_conversions(std::move(cfg)))) {
                return signalPassFailure();
            }
        }

        void runOnOperation() override { run_on_operation(); }
    };
} // namespace vast
