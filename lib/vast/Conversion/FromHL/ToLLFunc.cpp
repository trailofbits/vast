// Copyright (c) 2021-present, Trail of Bits, Inc.
#include "vast/Dialect/HighLevel/Passes.hpp"

#include "PassesDetails.hpp"

#include "vast/Conversion/Common/Passes.hpp"
#include "vast/Conversion/Common/Patterns.hpp"

#include "vast/Util/Common.hpp"
#include "vast/Util/DialectConversion.hpp"

namespace vast
{
    namespace pattern
    {
    } // namespace pattern

    struct HLToLLFunc : ModuleConversionPassMixin< HLToLLFunc, HLToLLFuncBase > {
        using base = ModuleConversionPassMixin< HLToLLFunc, HLToLLFuncBase >;

        static conversion_target create_conversion_target(mcontext_t &context) {
            conversion_target target(context);
            // TODO
            return target;
        }

        static void populate_conversions(config_t &config) {
            base::populate_conversions_base<
                // TODO
            >(config);
        }
    };
} // namespace vast


std::unique_ptr< mlir::Pass > vast::createHLToLLFuncPass()
{
    return std::make_unique< vast::HLToLLFunc>();
}
