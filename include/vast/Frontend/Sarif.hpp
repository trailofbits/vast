// Copyright (c) 2024, Trail of Bits, Inc.

#pragma once


#ifdef VAST_ENABLE_SARIF
    #include <gap/sarif/sarif.hpp>

    #include "vast/Frontend/Options.hpp"
    #include "vast/Util/Common.hpp"

namespace vast::cc::sarif {

    gap::sarif::location mk_location(loc_t loc);
    gap::sarif::location mk_location(file_loc_t loc);
    gap::sarif::location mk_location(name_loc_t loc);

    gap::sarif::physical_location get_physical_loc(file_loc_t loc);

    gap::sarif::level get_severity_level(mlir::DiagnosticSeverity severity);

    struct diagnostics {
        gap::sarif::run run;

        explicit diagnostics(const vast_args &vargs);

        auto handler() {
            return [&] (auto &diag) {
                gap::sarif::result result = {
                    .ruleId = "mlir-diag",
                    .message = { .text = diag.str() }
                };

                if (auto loc = mk_location(diag.getLocation()); loc.physicalLocation.has_value()) {
                    result.locations.push_back(std::move(loc));
                }

                result.level = get_severity_level(diag.getSeverity());
                run.results.push_back(std::move(result));
            };
        };

        gap::sarif::root emit(logical_result result) && {
            run.invocations[0].executionSuccessful = mlir::succeeded(result);
            return {
                .version = gap::sarif::version::k2_1_0,
                .runs{ std::move(run) },
            };
        }
    };

} // namespace vast::cc::sarif
#endif // VAST_ENABLE_SARIF
