// Copyright (c) 2023-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <llvm/ADT/StringMap.h>
VAST_UNRELAX_WARNINGS

#include "vast/Util/Pipeline.hpp"

namespace vast::cc {

    struct pipelines_config {
        llvm::StringMap< pipeline_step_builder > pipelines;
    };

    pipelines_config default_pipelines_config();

} // namespace vast::cc
