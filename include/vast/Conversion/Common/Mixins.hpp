// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include "mlir/Transforms/DialectConversion.h"
VAST_UNRELAX_WARNINGS

#include "vast/Conversion/Common/Patterns.hpp"
#include "vast/Conversion/Common/Types.hpp"
#include "vast/Conversion/TypeConverters/LLVMTypeConverter.hpp"

#include "vast/Util/TypeList.hpp"

namespace vast {

    template< typename self >
    struct populate_patterns
    {
        template< typename conversion >
        requires ( !util::is_type_list_v< conversion > )
        static void populate_conversions_impl(auto &cfg) {
            self::template add_pattern< conversion >(cfg);
            self::template legalize< conversion >(cfg);
        }

        template< typename list >
        requires util::is_type_list_v< list >
        static void populate_conversions_impl(auto &cfg) {
            if constexpr (!list::empty) {
                populate_conversions_impl< typename list::head >(cfg);
                self::template populate_conversions_impl< typename list::tail >(cfg);
            }
        }

        template< typename pattern >
        static void legalize(auto &cfg) {
            if constexpr (has_legalize<pattern>) {
                pattern::legalize(cfg.target);
            }
        }

        auto &underlying() { return static_cast<self &>(*this); }

        auto apply_conversions(auto &&cfg) {
            return mlir::applyPartialConversion(
                underlying().getOperation(), cfg.target, std::move(cfg.patterns)
            );
        }

        template< typename... conversions >
        static void populate_conversions(auto &cfg) {
            (self::template populate_conversions_impl< conversions >(cfg), ...);
        }
    };

    using lower_to_llvm_options = mlir::LowerToLLVMOptions;

    template< typename T >
    concept has_populate_conversions = requires(T a) { a.populate_conversions(); };

    template< typename T >
    concept has_run_after_conversion = requires(T a) { a.run_after_conversion(); };

    template< typename T >
    concept has_lower_to_llvm_options = requires(T a) {
        T::set_lower_to_llvm_options(std::declval< lower_to_llvm_options & >());
    };

    using rewrite_pattern_set = mlir::RewritePatternSet;

    // base configuration class
    struct base_conversion_config {
        rewrite_pattern_set patterns;
        conversion_target target;

        template< typename pattern >
        void add_pattern() {
            patterns.template add< pattern >(patterns.getContext());
        }
    };


    template< typename type_converter >
    struct type_converting_conversion_config : base_conversion_config {
        type_converter &tc;

        type_converting_conversion_config(
            rewrite_pattern_set patterns,
            conversion_target target,
            type_converter &tc
        )
            : base_conversion_config{std::move(patterns), std::move(target)}, tc(tc)
        {}

        template< typename pattern >
        void add_pattern() {
            patterns.template add< pattern >(tc, patterns.getContext());
        }
    };

    // Configuration class for LLVM conversion
    using llvm_type_converter = conv::tc::FullLLVMTypeConverter;

    struct llvm_conversion_config : base_conversion_config {
        llvm_type_converter &tc;

        llvm_conversion_config(
            rewrite_pattern_set patterns,
            conversion_target target,
            llvm_type_converter &tc
        )
            : base_conversion_config{std::move(patterns), std::move(target)}, tc(tc)
        {}

        llvm_conversion_config(llvm_conversion_config &&other)
            : base_conversion_config{std::move(other.patterns), std::move(other.target)}, tc(other.tc)
        {}

        template< typename pattern >
        void add_pattern() {
            patterns.template add< pattern >(tc);
        }
    };

    //
    // base class for module conversion passes providing common functionality
    //
    template< typename derived, template< typename > typename base >
    struct ConversionPassMixinBase
        : base< derived >
        , populate_patterns< derived >
    {
        using base_type = base<derived>;
        using patterns = populate_patterns<derived>;

        using base_type::getContext;
        using base_type::getOperation;
        using base_type::signalPassFailure;

        auto &self() { return static_cast< derived & >(*this); }

        template< typename pattern >
        static void add_pattern(auto &cfg) {
            cfg.template add_pattern< pattern >();
        }

        logical_result run_on_operation(auto &&cfg) {
            if (mlir::failed(patterns::apply_conversions(std::move(cfg)))) {
                return signalPassFailure(), mlir::failure();
            }
            return mlir::success();
        }

        logical_result run_on_operation() {
            auto cfg = self().make_config();
            self().populate_conversions(cfg);
            return run_on_operation(std::move(cfg));
        }

        void runOnOperation() override {
            if (mlir::succeeded(self().run_on_operation())) {
                if constexpr (has_run_after_conversion< derived >) {
                    self().run_after_conversion();
                }
            }
        }
    };

    //
    // Mixin to define simple module conversion passes. It requires the derived
    // pass to define static methods for specifying conversion targets
    // and populating rewrite patterns.
    //
    // To specify the legalization and illegalization of operations:
    //
    // `static conversion_target create_conversion_target(mcontext_t &context)`
    //
    // To populate rewrite patterns:
    //
    // `static void populate_conversions(base_conversion_config &cfg)`
    //
    // Example usage:
    //
    // struct ExamplePass : ConversionPassMixin<ExamplePass, ExamplePassBase> {
    //     using base = ConversionPassMixin<ExamplePass, ExamplePassBase>;
    //
    //     static conversion_target create_conversion_target(mcontext_t &context) {
    //         conversion_target target(context);
    //         // setup target here
    //         return target;
    //     }
    //
    //     static void populate_conversions(base_conversion_config &cfg) {
    //         base::populate_conversions<
    //             // pass conversion type_lists here
    //         >(cfg);
    //     }
    // }
    //
    template< typename derived, template< typename > typename base >
    struct ConversionPassMixin : ConversionPassMixinBase< derived, base >
    {
        base_conversion_config make_config() {
            auto &ctx = this->getContext();
            return { rewrite_pattern_set(&ctx), derived::create_conversion_target(ctx) };
        }
    };

    template< typename derived, template< typename > typename base, typename type_converter >
    struct TypeConvertingConversionPassMixin : ConversionPassMixinBase< derived, base >
    {
        std::shared_ptr< type_converter > tc;

        type_converting_conversion_config< type_converter > make_config() {
            auto &ctx = this->getContext();
            tc = std::make_shared< type_converter >(ctx);
            return { rewrite_pattern_set(&ctx), derived::create_conversion_target(ctx, *tc), *tc };
        }
    };

    //
    // Mixin for conversion passes that target the LLVM dialect.
    // Requires the derived pass to specify LLVM-specific conversion logic.
    //
    // Example usage:
    //
    // struct ExamplePass : LLVMConversionPassMixin<ExamplePass, ExamplePassBase> {
    //     using base = LLVMConversionPassMixin<ExamplePass, ExamplePassBase>;
    //
    //     static conversion_target create_conversion_target(mcontest_t &context) {
    //         conversion_target target(context);
    //         // setup target here
    //         return target;
    //     }
    //
    //     static void populate_conversions(llvm_conversion_config &patterns) {
    //         base::populate_conversions<
    //             // pass conversion type_lists here
    //         >(cfg);
    //     }
    // }
    //
    template< typename derived, template< typename > typename base >
    struct LLVMConversionPassMixin
        : ConversionPassMixinBase< derived, base >
    {
        std::shared_ptr< llvm_type_converter > tc;

        llvm_conversion_config make_config() {
            auto &ctx = this->getContext();
            const auto &dl_analysis = this->template getAnalysis< mlir::DataLayoutAnalysis >();

            lower_to_llvm_options llvm_options{&ctx};
            if constexpr (has_lower_to_llvm_options< derived >) {
                derived::set_lower_to_llvm_options(llvm_options);
            }

            tc = std::make_unique< llvm_type_converter >(
                this->getOperation(), &ctx, llvm_options, &dl_analysis
            );

            return { rewrite_pattern_set(&ctx), derived::create_conversion_target(ctx, tc), *tc };
        }
    };

} // namespace vast
