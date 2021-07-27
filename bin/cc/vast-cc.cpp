// Copyright (c) 2021-present, Trail of Bits, Inc.

#include "mlir/Support/LogicalResult.h"
#include "mlir/Translation.h"

#include "vast/Frontend/Parser.hpp"

int main(int argc, char **argv)
{
    // vast::registerAllTranslations();

    return failed(
        mlir::mlirTranslateMain(argc, argv, "VAST Translation Testing Tool")
    );
}