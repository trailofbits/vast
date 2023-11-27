// Copyright (c) 2023-present, Trail of Bits, Inc.

#include "vast/Frontend/Pipelines.hpp"

#include "vast/Dialect/HighLevel/Passes.hpp"

namespace vast::cc {

    pipelines_config default_pipelines_config() {
        return pipelines_config{{
            { "canonicalize", hl::make_canonicalize_pipeline },
        }};
    }

} // namespace vast::cc
