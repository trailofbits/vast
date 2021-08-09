// Copyright (c) 2021-present, Trail of Bits, Inc.

#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/MlirOptMain.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

#include "vast/Dialect/HighLevel/IR/HighLevel.hpp"
#include "vast/Dialect/VastDialect.hpp"

int main(int argc, char **argv)
{
    mlir::registerAllPasses();
    // Register VAST passes here

    mlir::DialectRegistry registry;
    registry.insert< vast::hl::HighLevelDialect >();
    registry.insert< mlir::StandardOpsDialect >();

    mlir::registerAllDialects(registry);
    return failed(
        mlir::MlirOptMain(argc, argv, "VAST Optimizer driver\n", registry)
    );
}