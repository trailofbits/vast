// Copyright (c) 2021-present, Trail of Bits, Inc.

#include "vast/Dialect/HighLevel/Passes.hpp"

#include "PassesDetails.hpp"

#include "vast/Util/Common.hpp"
#include "vast/Conversion/Common/Mixins.hpp"

namespace vast::hl {

    namespace pattern {

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
                mlir_value val = bld.create< hl::ConstantOp >(
                    ref.getLoc(), ref.getType(), ec.getValue()
                );

                ref.replaceAllUsesWith(val);
                ref->erase();

                return mlir::success();
            }
        };

    } // namespace pattern

    struct LowerEnumRefsPass : ConversionPassMixin< LowerEnumRefsPass, LowerEnumRefsBase >
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

} // namespace vast::hl

std::unique_ptr< mlir::Pass > vast::hl::createLowerEnumRefsPass() {
    return std::make_unique< vast::hl::LowerEnumRefsPass >();
}
