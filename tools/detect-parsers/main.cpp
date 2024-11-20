// Copyright (c) 2021-present, Trail of Bits, Inc.

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include "mlir/Conversion/Passes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
VAST_UNRELAX_WARNINGS

#include "vast/Conversion/Parser/Passes.hpp"
#include "vast/Dialect/Dialects.hpp"

#include "vast/Dialect/Parser/Dialect.hpp"

int main(int argc, char **argv)
{
    mlir::DialectRegistry registry;
    // register dialects
    vast::registerAllDialects(registry);
    mlir::registerAllDialects(registry);

    vast::registerParserConversionPasses();
    mlir::registerConversionPasses();
    registry.insert< vast::pr::ParserDialect >();

    return mlir::asMainReturnCode(
        mlir::MlirOptMain(argc, argv, "VAST Parser Detection driver\n", registry)
    );
}
