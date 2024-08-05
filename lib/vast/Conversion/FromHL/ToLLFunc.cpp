// Copyright (c) 2021-present, Trail of Bits, Inc.
#include "vast/Dialect/HighLevel/Passes.hpp"

#include "PassesDetails.hpp"

#include "vast/Conversion/Common/Passes.hpp"
#include "vast/Conversion/Common/Patterns.hpp"

#include "vast/Dialect/Core/Func.hpp"
#include "vast/Dialect/LowLevel/LowLevelOps.hpp"

#include "vast/Util/Common.hpp"
#include "vast/Util/DialectConversion.hpp"

namespace vast::conv::hltollfunc
{
    namespace pattern
    {
        struct func_op : operation_conversion_pattern< hl::FuncOp >
        {
            using base = operation_conversion_pattern< hl::FuncOp >;
            using base::base;
            using adaptor_t = hl::FuncOp::Adaptor;

            logical_result matchAndRewrite(
                hl::FuncOp op, adaptor_t adaptor, conversion_rewriter &rewriter) const override
            {
                return core::convert_and_replace_function< ll::FuncOp >(op, rewriter);
            }

            static void legalize(conversion_target &target) {
                target.addLegalOp< ll::FuncOp >();
                target.addIllegalOp< hl::FuncOp >();
            }

        };
    } // namespace pattern

    struct HLToLLFunc : ModuleConversionPassMixin< HLToLLFunc, HLToLLFuncBase > {
        using base   = ModuleConversionPassMixin< HLToLLFunc, HLToLLFuncBase >;
        using config = base::config;

        static conversion_target create_conversion_target(mcontext_t &context) {
            conversion_target target(context);
            return target;
        }

        static void populate_conversions(config &cfg) {
            base::populate_conversions_base<
                util::type_list< pattern::func_op >
            >(cfg);
        }
    };
} // namespace vast::conv::hltollfunc


std::unique_ptr< mlir::Pass > vast::createHLToLLFuncPass()
{
    return std::make_unique< vast::conv::hltollfunc::HLToLLFunc>();
}
