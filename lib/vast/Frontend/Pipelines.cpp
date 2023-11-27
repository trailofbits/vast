// Copyright (c) 2023-present, Trail of Bits, Inc.

#include "vast/Frontend/Pipelines.hpp"

#include "vast/Dialect/HighLevel/Passes.hpp"

namespace vast::cc {

    pipelines_config default_pipelines_config() {
        return pipelines_config{{
            { "canonicalize", hl::pipeline::canonicalize },
            { "desugar", hl::pipeline::desugar },
            { "simplify", hl::pipeline::simplify },
            { "stdtypes", hl::pipeline::stdtypes }
        }};
    }

} // namespace vast::cc
