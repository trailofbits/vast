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
    //  Since we cannot easily do `using config = self::config` the type is instead
    //  taken as a template.
    template< typename self >
    struct populate_patterns
    {
        template< typename list, typename config >
        static void populate_conversions_impl(config &cfg) {
            if constexpr (!list::empty) {
                self::template add_pattern< typename list::head >(cfg);
                self::template legalize< typename list::head >(cfg);
                return self::template populate_conversions_impl< typename list::tail >(cfg);
            }
        }

        template< typename pattern, typename config >
        static void legalize(config &cfg) {
            if constexpr (has_legalize< pattern >) {
                pattern::legalize(cfg.target);
            }
        }

        auto &underlying() { return static_cast< self & >(*this); }

        // It is expected to move into this method, as it consumes & runs a configuration.
        template< typename config >
        auto apply_conversions(config cfg) {
            return mlir::applyPartialConversion(
                underlying().getOperation(), cfg.target, std::move(cfg.patterns)
            );
        }

        template< typename... lists, typename config >
        static void populate_conversions_base(config &cfg) {
            (self::template populate_conversions_impl< lists >(cfg), ...);
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
    // `static void populate_conversions(config &cfg)`
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
    template< typename derived, template< typename > typename base >
    struct ModuleConversionPassMixin
        : base< derived >
        , populate_patterns< derived >
    {
        using base_type = base< derived >;
        using populate  = populate_patterns< derived >;

        using base_type::getContext;
        using base_type::getOperation;
        using base_type::signalPassFailure;

        using rewrite_pattern_set = mlir::RewritePatternSet;

        struct config
        {
            rewrite_pattern_set patterns;
            conversion_target target;

            mlir::MLIRContext *getContext() { return patterns.getContext(); }
        };

        template< typename pattern >
        static void add_pattern(config &config) {
            config.patterns.template add< pattern >(config.getContext());
        }

        auto &self() { return static_cast< derived & >(*this); }

        void populate_conversions(config &) {}

        logical_result run_on_operation() {
            auto &ctx = getContext();
            config cfg = {
                rewrite_pattern_set(&ctx), derived::create_conversion_target(ctx)
            };

            self().populate_conversions(cfg);

            if (mlir::failed(populate::apply_conversions(std::move(cfg)))) {
                return signalPassFailure(), mlir::failure();
            }

            return mlir::success();
        }

        void runOnOperation() override {
            if (mlir::succeeded(run_on_operation())) {
                this->after_operation();
            }
        }

        // Override to specify what is supposed to run after `run_on_operation` is finished.
        // This will run *only if the `run_on_operation* was successful.
        virtual void after_operation() {};
    };

    // Sibling of the above module for passes that go to the LLVM dialect.
    // Example usage:
    //
    // struct ExamplePass : ModuleLLVMConversionPassMixin< ExamplePass, ExamplePassBase > {
    //     using base_type = ModuleLLVMConversionPassMixin< ExamplePass, ExamplePassBase >;
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
    template< typename derived, template< typename > typename base >
    struct ModuleLLVMConversionPassMixin
        : base< derived >
        , populate_patterns< derived >
    {
        using base_type = base< derived >;
        using populate  = populate_patterns< derived >;

        using base_type::getContext;
        using base_type::getOperation;
        using base_type::signalPassFailure;

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

        auto &self() { return static_cast< derived & >(*this); }

        void populate_conversions(config &) {}

        template< typename pattern >
        static void add_pattern(config &cfg) {
            cfg.patterns.template add< pattern >(cfg.tc);
        }

        logical_result run_on_operation() {
            auto &ctx               = getContext();
            const auto &dl_analysis = this->template getAnalysis< mlir::DataLayoutAnalysis >();

            mlir::LowerToLLVMOptions llvm_options{ &ctx };
            derived::set_llvm_opts(llvm_options);

            auto tc  = llvm_type_converter(getOperation(), &ctx, llvm_options, &dl_analysis);
            config cfg = {
                rewrite_pattern_set(&ctx), derived::create_conversion_target(ctx, tc), tc
            };

            // populate all patterns
            self().populate_conversions(cfg);

            if (failed(populate::apply_conversions(std::move(cfg)))) {
                return signalPassFailure(), mlir::failure();
            }

            return mlir::success();
        }

        void runOnOperation() override {
            if (mlir::succeeded(run_on_operation())) {
                this->after_operation();
            }
        }

        virtual void after_operation() {};
    };

} // namespace vast
