// Copyright (c) 2024, Trail of Bits, Inc.

#include "vast/Frontend/Sarif.hpp"

#ifdef VAST_ENABLE_SARIF

#include "vast/Config/config.h"

namespace vast::cc::sarif {

    gap::sarif::location mk_location(loc_t loc) {
        if (auto file_loc = mlir::dyn_cast< file_loc_t >(loc)) {
            return mk_location(file_loc);
        } else if (auto name_loc = mlir::dyn_cast< name_loc_t >(loc)) {
            return mk_location(name_loc);
        }

        return {};
    }

    gap::sarif::location mk_location(file_loc_t loc) {
        return {
            .physicalLocation{ get_physical_loc(loc) },
        };
    }

    gap::sarif::location mk_location(name_loc_t loc) {
        return {
            .physicalLocation{
                get_physical_loc(mlir::cast< file_loc_t >(loc.getChildLoc()))
            },
            .logicalLocations{ { .name = loc.getName().str() } },
        };
    }

    gap::sarif::physical_location get_physical_loc(file_loc_t loc) {
        std::filesystem::path file_path{ loc.getFilename().str() };
        auto abs_path = std::filesystem::absolute(file_path);
        return {
            .artifactLocation{ { .uri{ "file://" + abs_path.string() } } },
            .region{ {
                .startLine   = loc.getLine(),
                .startColumn = loc.getColumn(),
            } },
        };
    }

    gap::sarif::level get_severity_level(mlir::DiagnosticSeverity severity) {
        using enum gap::sarif::level;
        using enum mlir::DiagnosticSeverity;
        switch (severity) {
            case Note: return kNote;
            case Warning: return kWarning;
            case Error: return kError;
            case Remark: return kNote;
        }
    }

    diagnostics::diagnostics(const vast_args &vargs)
        : run(gap::sarif::run{
            .tool{
                .driver{
                    .name           = "vast-front",
                    .organization   = "Trail of Bits, inc.",
                    .product        = "VAST",
                    .version        = std::string{ vast::version },
                    .informationUri = std::string{ vast::homepage_url },
                },
            },
            .invocations{{
                .arguments{ std::begin(vargs.args), std::end(vargs.args) },
                .executionSuccessful = true,
            }}
        })
    {}

} // namespace vast::cc::sarif

#endif // VAST_ENABLE_SARIF
