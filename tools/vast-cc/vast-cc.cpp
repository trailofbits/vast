// Copyright (c) 2021-present, Trail of Bits, Inc.

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <clang/AST/ASTConsumer.h>
#include <clang/AST/RecursiveASTVisitor.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Frontend/FrontendAction.h>
#include <clang/Tooling/Tooling.h>
#include <clang/Tooling/CommonOptionsParser.h>
#include <mlir/Tools/mlir-translate/MlirTranslateMain.h>
VAST_UNRELAX_WARNINGS

#include <vast/CodeGen/Register.hpp>

int main(int argc, char **argv) {
    vast::registerAllTranslations();

    return failed(
        mlir::mlirTranslateMain(argc, argv, "VAST Translation Testing Tool")
    );
}
