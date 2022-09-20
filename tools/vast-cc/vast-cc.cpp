// Copyright (c) 2021-present, Trail of Bits, Inc.

#include <vast/Translation/Register.hpp>

#include <mlir/Support/LogicalResult.h>
#include <mlir/Tools/mlir-translate/MlirTranslateMain.h>

int main(int argc, char **argv)
{
    vast::registerAllTranslations();

    return failed(
        mlir::mlirTranslateMain(argc, argv, "VAST Translation Testing Tool")
    );
}
