// Copyright (c) 2021-present, Trail of Bits, Inc.

#include "vast/Util/Functions.hpp"
#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <clang/Frontend/ASTConsumers.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Frontend/FrontendAction.h>
#include <clang/Tooling/Tooling.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>
#include <mlir/Dialect/DLTI/DLTI.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Tools/mlir-translate/Translation.h>
VAST_UNRELAX_WARNINGS

#include "vast/Dialect/HighLevel/HighLevelDialect.hpp"
#include "vast/Dialect/HighLevel/HighLevelAttributes.hpp"
#include "vast/Dialect/HighLevel/HighLevelTypes.hpp"
#include "vast/Translation/CodeGen.hpp"
#include "vast/Util/Common.hpp"

namespace vast::hl
{
    static llvm::cl::list< std::string > compiler_args(
        "ccopts", llvm::cl::ZeroOrMore, llvm::cl::desc("Specify compiler options")
    );

    static llvm::cl::opt< bool > id_meta_flag(
        "id-meta", llvm::cl::desc("Attach ids to nodes as metadata")
    );

    static OwningModuleRef from_source_parser(
        const llvm::MemoryBuffer *input, mlir::MLIRContext *mctx
    ) {
        auto ast = clang::tooling::buildASTFromCodeWithArgs(
            input->getBuffer(), compiler_args
        );

        auto actx = &ast->getASTContext();

        if (id_meta_flag) {
            return CodeGenWithMetaIDs(actx, mctx).emit_module(ast.get());
        } else {
            return DefaultCodeGen(actx, mctx).emit_module(ast.get());
        }
    }

    mlir::LogicalResult registerFromSourceParser() {
        mlir::TranslateToMLIRRegistration from_source(
            "from-source",
            [](llvm::SourceMgr &mgr, mlir::MLIRContext *ctx) -> OwningModuleRef {
                VAST_CHECK(mgr.getNumBuffers() == 1,    "expected single input buffer");
                auto buffer = mgr.getMemoryBuffer(mgr.getMainFileID());
                return from_source_parser(buffer, ctx);
            });

        return mlir::success();
    }

} // namespace vast::hl
