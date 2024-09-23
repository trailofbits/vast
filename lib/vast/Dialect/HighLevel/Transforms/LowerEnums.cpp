// Copyright (c) 2021-present, Trail of Bits, Inc.

#include "vast/Dialect/HighLevel/Passes.hpp"

#include "PassesDetails.hpp"

#include "vast/Util/Common.hpp"
#include "vast/Conversion/Common/Mixins.hpp"
#include "vast/Conversion/Common/Patterns.hpp"
#include "vast/Conversion/TypeConverters/TypeConvertingPattern.hpp"

namespace vast::hl {

    struct EnumTypeConverter
        : conv::tc::identity_type_converter
        , conv::tc::mixins< EnumTypeConverter >
        , conv::tc::function_type_converter< EnumTypeConverter >
    {
        mcontext_t &mctx;

        explicit EnumTypeConverter(mcontext_t &mctx)
            : mctx(mctx)
        {
            conv::tc::function_type_converter< EnumTypeConverter >::init();
        }
    };

    namespace pattern {

        using lower_enum_types = conv::tc::generic_type_converting_pattern<
            EnumTypeConverter
        >;

        struct erase_enum_decl : operation_conversion_pattern< hl::EnumDeclOp > {
            using base = operation_conversion_pattern< hl::EnumDeclOp >;
            using base::base;

            using adaptor_t = hl::EnumDeclOp::Adaptor;

            logical_result matchAndRewrite(
                hl::EnumDeclOp decl, adaptor_t adaptor, conversion_rewriter &rewriter
            ) const override {
                rewriter.eraseOp(decl);
                return mlir::success();
            }
        };

        struct erase_enum_ref : operation_conversion_pattern< hl::EnumRefOp > {
            using base = operation_conversion_pattern< hl::EnumRefOp >;
            using base::base;

            using adaptor_t = hl::EnumRefOp::Adaptor;

            logical_result matchAndRewrite(
                hl::EnumRefOp ref, adaptor_t adaptor, conversion_rewriter &rewriter
            ) const override {
                auto op = core::symbol_table::lookup< core::enum_constant_symbol >(
                    ref, ref.getName()
                );

                auto ec = mlir::dyn_cast< hl::EnumConstantOp >(op);
                VAST_CHECK(ec, "Enum constant symbol is not anhl::EnumConstantOp.");

                auto bld = mlir::OpBuilder(ref);
                auto con = bld.create< hl::ConstantOp >(
                    ref.getLoc(), ref.getType(), ec.getValue()
                );

                rewriter.replaceOp(ref, con);
                return mlir::success();
            }
        };

    } // namespace pattern

    struct LowerEnumRefsPass
        : ConversionPassMixin< LowerEnumRefsPass, LowerEnumRefsBase >
    {
        using base = ConversionPassMixin< LowerEnumRefsPass, LowerEnumRefsBase >;

        static conversion_target create_conversion_target(mcontext_t &mctx) {
            auto trg = conversion_target(mctx);
            trg.addIllegalOp< hl::EnumRefOp >();
            return trg;
        }

        static void populate_conversions(base_conversion_config &cfg) {
            base::populate_conversions< pattern::erase_enum_ref >(cfg);
        }
    };

    struct LowerEnumDeclsPass
        : TypeConvertingConversionPassMixin<
            LowerEnumDeclsPass,
            LowerEnumDeclsBase,
            EnumTypeConverter
        >
    {
        using base = TypeConvertingConversionPassMixin<
            LowerEnumDeclsPass, LowerEnumDeclsBase, EnumTypeConverter
        >;

        static conversion_target create_conversion_target(mcontext_t &mctx, auto &tc) {
            auto trg = conversion_target(mctx);
            return trg;
        }

        static void populate_conversions(base_conversion_config &cfg) {
            base::populate_conversions< pattern::lower_enum_types >(cfg);
            base::populate_conversions< pattern::erase_enum_decl >(cfg);
        }
    };
} // namespace vast::hl

std::unique_ptr< mlir::Pass > vast::hl::createLowerEnumRefsPass() {
    return std::make_unique< vast::hl::LowerEnumRefsPass >();
}

std::unique_ptr< mlir::Pass > vast::hl::createLowerEnumDeclsPass() {
    return std::make_unique< vast::hl::LowerEnumDeclsPass >();
}
