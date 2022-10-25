// Copyright (c) 2021-present, Trail of Bits, Inc.

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <mlir/Support/LogicalResult.h>
#include <mlir/Tools/mlir-translate/MlirTranslateMain.h>
VAST_UNRELAX_WARNINGS

#include <vast/Translation/Register.hpp>

int main(int argc, char **argv)
{
    vast::registerAllTranslations();

    return failed(
        mlir::mlirTranslateMain(argc, argv, "VAST Translation Testing Tool")
    );
}
