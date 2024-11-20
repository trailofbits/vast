// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
VAST_UNRELAX_WARNINGS

#include <gap/sarif/sarif.hpp>

namespace vast {
    struct ParserSourceDetector
        : mlir::PassWrapper< ParserSourceDetector, mlir::OperationPass< mlir::ModuleOp > >
    {
        std::vector< gap::sarif::result > &results;

        ParserSourceDetector(std::vector< gap::sarif::result > &results) : results(results) {}

        void runOnOperation() override;
    };
} // namespace vast
