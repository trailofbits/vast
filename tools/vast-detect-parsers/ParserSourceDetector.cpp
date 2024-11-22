// Copyright (c) 2021-present, Trail of Bits, Inc.

#ifdef VAST_ENABLE_SARIF
    #include "SarifPasses.hpp"

    #include "vast/Dialect/Parser/Ops.hpp"
    #include "vast/Frontend/Sarif.hpp"

namespace vast {
    void ParserSourceDetector::runOnOperation() {
        getOperation().walk([&](pr::Source op) {
            gap::sarif::result result{
                .ruleId{ "pr-source" },
                .ruleIndex = 0,
                .kind      = gap::sarif::kind::kInformational,
                .level     = gap::sarif::level::kNote,
                .message{
                        .text{ { "Parser source detected" } },
                        },
                .locations{},
            };
            if (auto loc = cc::sarif::mk_location(op.getLoc());
                loc.physicalLocation.has_value())
            {
                result.locations.push_back(std::move(loc));
            }
            results.push_back(result);
        });
    }
} // namespace vast
#endif
