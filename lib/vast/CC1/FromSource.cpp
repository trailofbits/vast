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
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Tools/mlir-translate/Translation.h>
#include <mlir/Pass/PassManager.h>
VAST_UNRELAX_WARNINGS

#include "vast/Dialect/HighLevel/HighLevelDialect.hpp"
#include "vast/Dialect/HighLevel/HighLevelAttributes.hpp"
#include "vast/Dialect/HighLevel/HighLevelTypes.hpp"
#include "vast/Dialect/HighLevel/Passes.hpp"
#include "vast/CodeGen/CodeGen.hpp"
#include "vast/Util/Common.hpp"

namespace vast::hl
{
    static llvm::cl::list< std::string > compiler_args(
        "ccopts", llvm::cl::ZeroOrMore, llvm::cl::desc("Specify compiler options")
    );

    static llvm::cl::opt< bool > id_meta_flag(
        "id-meta", llvm::cl::desc("Attach ids to nodes as metadata")
    );

    static llvm::cl::opt< bool > disable_splicing(
        "disable-scope-splicing-pass", llvm::cl::desc("Disable pass that splices trailing scopes (useful for debugging purposes)")
    );

    static owning_module_ref from_source_parser(
        const llvm::MemoryBuffer *input, mcontext_t *mctx
    ) {
        auto ast = clang::tooling::buildASTFromCodeWithArgs(
            input->getBuffer(), compiler_args
        );

        auto actx = &ast->getASTContext();

        cg::CodeGenContext cgctx(*mctx, *actx);

        if (id_meta_flag) {
            cg::CodeGenWithMetaIDs(cgctx).emit_module(ast.get());
        } else {
            cg::DefaultCodeGen(cgctx).emit_module(ast.get());
        }

        return std::move(cgctx.mod);
    }

    mlir::LogicalResult registerFromSourceParser() {
        mlir::TranslateToMLIRRegistration from_source(
            "from-source", "from c/c++ source code to vast high-level mlir",
            [](llvm::SourceMgr &mgr, mlir::MLIRContext *ctx) -> owning_module_ref {
                VAST_CHECK(mgr.getNumBuffers() == 1,    "expected single input buffer");
                auto buffer = mgr.getMemoryBuffer(mgr.getMainFileID());

                auto mod = from_source_parser(buffer, ctx);

                if (!disable_splicing) {
                    mlir::PassManager pass_mgr(ctx);
                    pass_mgr.addPass(hl::createSpliceTrailingScopes());
                    VAST_ASSERT(pass_mgr.run(mod.get()).succeeded());
                }

                return mod;
            });

        return mlir::success();
    }

} // namespace vast::hl
