// Copyright (c) 2021-present, Trail of Bits, Inc.

#include "vast/Dialect/VastOps.hpp"
#include <mlir/IR/Location.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Translation.h>
#include <mlir/IR/Builders.h>
#include <mlir/Support/LogicalResult.h>

#include <clang/AST/RecursiveASTVisitor.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Frontend/ASTConsumers.h>
#include "clang/AST/Decl.h"
#include "clang/AST/Stmt.h"
#include "clang/Frontend/FrontendAction.h"
#include <clang/AST/ASTConsumer.h>
#include <clang/AST/DeclBase.h>
#include <clang/Tooling/Tooling.h>

#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/Debug.h>

#include <vast/Dialect/VastDialect.hpp>
#include <vast/Dialect/VastOps.hpp>


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

        ASTVisitor(mlir::ModuleOp module)
            : module(module)
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

        bool VisitFunctionDecl(clang::FunctionDecl *fndecl)
        {
            LLVM_DEBUG(llvm::dbgs() << "Visit FunctionDecl: " << fndecl->getName() << "\n");

            Builder bld(module.getBodyRegion());
            auto loc = fileLineColLoc(fndecl);

            bld.create< VastFuncOp >(loc, fndecl->getName());

            return true;
        }

        bool VisitCompoundStmt(clang::CompoundStmt *stmt)
        {
            LLVM_DEBUG(llvm::dbgs() << "Visit CompoundStmt\n");
            return true;
        }

        mlir::Location fileLineColLoc(clang::Decl *decl)
        {
            auto loc = decl->getLocation();
            auto &mgr = decl->getASTContext().getSourceManager();

            auto name = mgr.getFilename(loc);
            auto line = mgr.getPresumedLineNumber(loc);
            auto col = mgr.getPresumedColumnNumber(loc);

            return mlir::FileLineColLoc::get(name, line, col, module->getContext());
        }

        mlir::ModuleOp module;
    };

    static mlir::OwningModuleRef from_source_parser(const llvm::MemoryBuffer *input, mlir::MLIRContext *ctx)
    {
        ctx->loadDialect< VastDialect >();

        mlir::OwningModuleRef module(
            mlir::ModuleOp::create(
                mlir::FileLineColLoc::get(input->getBufferIdentifier(), /* line */ 0, /* column */ 0, ctx)
            )
        );

        ASTVisitor visitor(*module);
        auto ast = clang::tooling::buildASTFromCode(input->getBuffer());
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