// Copyright (c) 2021-present, Trail of Bits, Inc.

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
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
VAST_UNRELAX_WARNINGS

#include "vast/Dialect/HighLevel/HighLevelDialect.hpp"
#include "vast/Dialect/HighLevel/Passes.hpp"

int main(int argc, char **argv)
{
    mlir::registerAllPasses();
    // Register VAST passes here
    vast::hl::registerPasses();

    mlir::DialectRegistry registry;
    registry.insert< vast::hl::HighLevelDialect >();
    registry.insert< mlir::StandardOpsDialect >();
    registry.insert< mlir::LLVM::LLVMDialect >();

    mlir::registerAllDialects(registry);
    return failed(
        mlir::MlirOptMain(argc, argv, "VAST Optimizer driver\n", registry)
    );
}
