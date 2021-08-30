// Copyright (c) 2021-present, Trail of Bits, Inc.

#include <mlir/IR/Location.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Translation.h>
#include <mlir/IR/Builders.h>
#include <mlir/Support/LogicalResult.h>

#include <clang/AST/RecursiveASTVisitor.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Frontend/ASTConsumers.h>
#include <clang/AST/Decl.h>
#include <clang/AST/Stmt.h>
#include <clang/Frontend/FrontendAction.h>
#include <clang/AST/ASTConsumer.h>
#include <clang/AST/DeclBase.h>
#include <clang/Tooling/Tooling.h>

#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/Debug.h>

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Identifier.h"
#include "vast/Translation/Types.hpp"
#include "vast/Dialect/HighLevel/HighLevel.hpp"
#include "vast/Dialect/HighLevel/HighLevelTypes.hpp"
#include "clang/AST/ASTContext.h"
#include "llvm/Support/ErrorHandling.h"

#include <iostream>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <fstream>

#define DEBUG_TYPE "vast-from-source"

namespace vast::hl
{
    struct ASTVisitor : clang::RecursiveASTVisitor<ASTVisitor>
    {
        using Builder = mlir::OpBuilder;

        ASTVisitor(mlir::ModuleOp module, clang::ASTContext &ast_ctx)
            : module(module), types(module->getContext()), ast_ctx(ast_ctx)
        {}

        virtual ~ASTVisitor() = default;

        virtual bool VisitDecl(clang::Decl *decl)
        {
            LLVM_DEBUG(llvm::dbgs() << "Visit Decl\n");
            return true;
        }

        virtual bool VisitStmt(clang::Stmt *stmt)
        {
            LLVM_DEBUG(llvm::dbgs() << "Visit Stmt\n");
            return true;
        }

        bool VisitTranslationUnitDecl(clang::TranslationUnitDecl *tu)
        {
            LLVM_DEBUG(llvm::dbgs() << "Visit Translation Unit\n");
            return true;
        }

        bool VisitTypedefDecl(clang::TypedefDecl *tdef)
        {
            LLVM_DEBUG(llvm::dbgs() << "Visit Typedef\n");
            return true;
        }

        bool VisitFunctionDecl(clang::FunctionDecl *decl)
        {
            LLVM_DEBUG(llvm::dbgs() << "Visit FunctionDecl: " << decl->getName() << "\n");

            Builder bld(module.getBodyRegion());

            auto loc = getLocation(decl->getSourceRange());
            auto type = types.convert(decl->getFunctionType());
            assert( type );

            auto fn = bld.create< mlir::FuncOp >(loc, decl->getName(), type);

            auto entry = fn.addEntryBlock();
            bld.setInsertionPointToStart(entry);

            // emit function body
            TraverseStmt(decl->getBody());

            // TODO(Heno): fix return generation
            bld.create< mlir::ReturnOp >(getLocation(decl->getEndLoc()));

            return true;
        }

        bool VisitCompoundStmt(clang::CompoundStmt *stmt)
        {
            LLVM_DEBUG(llvm::dbgs() << "Visit CompoundStmt\n");
            return true;
        }

        bool VisitReturnStmt(clang::ReturnStmt *stmt)
        {
            LLVM_DEBUG(llvm::dbgs() << "Visit ReturnStmt\n");
            return true;
        }

        mlir::Location getLocation(clang::SourceRange range)
        {
            auto ctx = module.getContext();

            auto beg = range.getBegin();
            auto loc = ast_ctx.getSourceManager().getPresumedLoc(beg);

            Builder bld(module.getBodyRegion());
            if (loc.isInvalid())
                return bld.getUnknownLoc();

            auto file = mlir::Identifier::get(loc.getFilename(), ctx);
            return bld.getFileLineColLoc(file, loc.getLine(), loc.getColumn());
        }

        mlir::ModuleOp module;
        TypeConverter types;

        const clang::ASTContext &ast_ctx;
    };

    static mlir::OwningModuleRef from_source_parser(const llvm::MemoryBuffer *input, mlir::MLIRContext *ctx)
    {
        ctx->loadDialect< HighLevelDialect >();
        ctx->loadDialect< mlir::StandardOpsDialect >();

        mlir::OwningModuleRef module(
            mlir::ModuleOp::create(
                mlir::FileLineColLoc::get(input->getBufferIdentifier(), /* line */ 0, /* column */ 0, ctx)
            )
        );

        auto ast = clang::tooling::buildASTFromCode(input->getBuffer());

        ASTVisitor visitor(*module, ast->getASTContext());
        visitor.TraverseDecl(ast->getASTContext().getTranslationUnitDecl());

        // TODO(Heno): verify module
        return module;
    }

    mlir::LogicalResult registerFromSourceParser()
    {
        mlir::TranslateToMLIRRegistration from_source( "from-source",
            [] (llvm::SourceMgr &mgr, mlir::MLIRContext *ctx) -> mlir::OwningModuleRef {
                assert(mgr.getNumBuffers() == 1 && "expected single input buffer");
                auto buffer = mgr.getMemoryBuffer(mgr.getMainFileID());
                return from_source_parser(buffer, ctx);
            }
        );

        return mlir::success();
    }

} // namespace vast::hl
