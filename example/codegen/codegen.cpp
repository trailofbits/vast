#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <clang/AST/ASTContext.h>
#include <clang/Tooling/Tooling.h>
#include <mlir/IR/Builders.h>
VAST_UNRELAX_WARNINGS

#include "vast/Translation/CodeGen.hpp"

#include <fstream>
#include <iostream>

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cerr << "usage ./codegen <input.cpp>\n";
        std::exit(1);
    }

    // load clang ast of source file
    std::ifstream ifs(argv[1]);
    std::string source(
        (std::istreambuf_iterator< char >(ifs)), (std::istreambuf_iterator< char >()));

    auto ast = clang::tooling::buildASTFromCode(source);

    // setup mlir environment
    mlir::DialectRegistry registry;
    registry.insert< vast::hl::HighLevelDialect, mlir::StandardOpsDialect, mlir::DLTIDialect >();
    mlir::MLIRContext ctx(registry);

    // generate ir for ast declaration
    vast::hl::high_level_codegen codegen(&ctx);

    auto &actx = ast->getASTContext();
    auto tu = actx.getTranslationUnitDecl();

    for (const auto &decl : tu->decls()) {
        if (auto fn = clang::dyn_cast< clang::FunctionDecl >(decl)) {
            llvm::errs() << "module for function: " << fn->getName() << "\n";
            auto ir = codegen.emit_module(fn);
            ir->dump();
        }
    }

}
