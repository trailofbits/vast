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
#include <mlir/Dialect/SCF/SCF.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Identifier.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Translation.h>
VAST_UNRELAX_WARNINGS

#include "vast/Dialect/HighLevel/HighLevelDialect.hpp"
#include "vast/Dialect/HighLevel/HighLevelTypes.hpp"
#include "vast/Translation/HighLevelVisitor.hpp"

namespace vast::hl
{

    struct VastCodeGen : clang::ASTConsumer {
        VastCodeGen(TranslationContext &ctx)
            : ctx(ctx) {}

        bool HandleTopLevelDecl(clang::DeclGroupRef) override { UNIMPLEMENTED; }

        void emit_data_layout(const dl::DataLayoutBlueprint &dl) {
            auto &mctx = ctx.getMLIRContext();
            std::vector< mlir::DataLayoutEntryInterface > entries;
            for (const auto &[_, e] : dl.entries)
                entries.push_back(e.wrap(mctx));
            ctx.getModule().get()->setAttr(
                mlir::DLTIDialect::kDataLayoutAttrName,
                mlir::DataLayoutSpecAttr::get(&mctx, entries));
        }

        void HandleTranslationUnit(clang::ASTContext &) override {
            auto tu = ctx.getASTContext().getTranslationUnitDecl();
            CodeGenVisitor visitor(ctx);

            for (const auto &decl : tu->decls())
                visitor.Visit(decl);

            // parform after we gather all types from the translation unit
            emit_data_layout(ctx.data_layout());
        }

      private:
        TranslationContext &ctx;
    };

    static llvm::cl::list< std::string > compiler_args(
        "ccopts", llvm::cl::ZeroOrMore, llvm::cl::desc("Specify compiler options"));

    static mlir::OwningModuleRef from_source_parser(
        const llvm::MemoryBuffer *input, mlir::MLIRContext *ctx) {
        ctx->loadDialect< HighLevelDialect >();
        ctx->loadDialect< mlir::StandardOpsDialect >();
        ctx->loadDialect< mlir::DLTIDialect >();
        ctx->loadDialect< mlir::scf::SCFDialect >();

        mlir::OwningModuleRef mod(mlir::ModuleOp::create(mlir::FileLineColLoc::get(
            ctx, input->getBufferIdentifier(), /* line */ 0, /* column */ 0)));

        auto ast = clang::tooling::buildASTFromCodeWithArgs(input->getBuffer(), compiler_args);

        TranslationContext tctx(*ctx, ast->getASTContext(), mod);

        // TODO(Heno): verify correct scopes of type names
        llvm::ScopedHashTableScope type_def_scope(tctx.type_defs);
        llvm::ScopedHashTableScope type_dec_scope(tctx.type_decls);
        llvm::ScopedHashTableScope enum_dec_scope(tctx.enum_decls);
        llvm::ScopedHashTableScope func_scope(tctx.functions);

        VastCodeGen codegen(tctx);
        codegen.HandleTranslationUnit(ast->getASTContext());

        // TODO(Heno): verify module
        return mod;
    }

    mlir::LogicalResult registerFromSourceParser() {
        mlir::TranslateToMLIRRegistration from_source(
            "from-source",
            [](llvm::SourceMgr &mgr, mlir::MLIRContext *ctx) -> mlir::OwningModuleRef {
                assert(mgr.getNumBuffers() == 1 && "expected single input buffer");
                auto buffer = mgr.getMemoryBuffer(mgr.getMainFileID());
                return from_source_parser(buffer, ctx);
            });

        return mlir::success();
    }

} // namespace vast::hl
