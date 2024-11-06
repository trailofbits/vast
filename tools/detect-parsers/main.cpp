// Copyright (c) 2021-present, Trail of Bits, Inc.

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
VAST_UNRELAX_WARNINGS

#include "vast/Conversion/Passes.hpp"

#include "vast/Conversion/Passes.hpp"
#include "vast/Dialect/Dialects.hpp"

int main(int argc, char **argv)
{
    mlir::DialectRegistry registry;
    // register dialects
    vast::registerAllDialects(registry);
    mlir::registerAllDialects(registry);

    return failed(
        mlir::MlirOptMain(argc, argv, "VAST Parser Detection driver\n", registry)
    );
}
