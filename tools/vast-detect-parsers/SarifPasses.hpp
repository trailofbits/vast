// Copyright (c) 2024, Trail of Bits, Inc.

#pragma once

#ifdef VAST_ENABLE_SARIF
    #include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
    #include <mlir/IR/BuiltinOps.h>
    #include <mlir/Pass/Pass.h>
    #include <mlir/Pass/PassManager.h>
VAST_UNRELAX_WARNINGS

    #include <gap/sarif/sarif.hpp>

namespace vast {
    struct ParserCategoryDetector
        : mlir::PassWrapper< ParserCategoryDetector, mlir::OperationPass< mlir::ModuleOp > >
    {
        std::vector< gap::sarif::result > &results;

        ParserCategoryDetector(std::vector< gap::sarif::result > &results) : results(results) {}

        void runOnOperation() override;
    };
} // namespace vast
#endif
