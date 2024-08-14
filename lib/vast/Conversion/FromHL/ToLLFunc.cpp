// Copyright (c) 2021-present, Trail of Bits, Inc.
#include "vast/Dialect/HighLevel/Passes.hpp"

#include "PassesDetails.hpp"

#include "vast/Conversion/Common/Mixins.hpp"
#include "vast/Conversion/Common/Patterns.hpp"

#include "vast/Dialect/Core/Func.hpp"
#include "vast/Dialect/LowLevel/LowLevelOps.hpp"

#include "vast/Util/Common.hpp"
#include "vast/Util/DialectConversion.hpp"

namespace vast::conv::hltollfunc
{
    namespace pattern
    {
        struct func_op : one_to_one_conversion_pattern< hl::FuncOp, ll::FuncOp >
        {
            using base = one_to_one_conversion_pattern< hl::FuncOp, ll::FuncOp >;
            using base::base;

            using adaptor_t = hl::FuncOp::Adaptor;

            logical_result matchAndRewrite(
                hl::FuncOp op, adaptor_t adaptor, conversion_rewriter &rewriter
            ) const override {
                return core::convert_and_replace_function< ll::FuncOp >(op, rewriter);
            }
        };
    } // namespace pattern

    struct HLToLLFunc : ModuleConversionPassMixin< HLToLLFunc, HLToLLFuncBase >
    {
        using base = ModuleConversionPassMixin< HLToLLFunc, HLToLLFuncBase >;

        static conversion_target create_conversion_target(mcontext_t &context) {
            return { context };
        }

        static void populate_conversions(auto &cfg) {
            base::populate_conversions< pattern::func_op >(cfg);
        }
    };
} // namespace vast::conv::hltollfunc


std::unique_ptr< mlir::Pass > vast::createHLToLLFuncPass()
{
    return std::make_unique< vast::conv::hltollfunc::HLToLLFunc>();
}
