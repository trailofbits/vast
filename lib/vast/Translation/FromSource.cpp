// Copyright (c) 2021-present, Trail of Bits, Inc.

#include <mlir/IR/Location.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Translation.h>
#include <mlir/Support/LogicalResult.h>

#include <clang/AST/RecursiveASTVisitor.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Frontend/ASTConsumers.h>
#include "clang/Frontend/FrontendAction.h"
#include <clang/AST/ASTConsumer.h>
#include <clang/AST/DeclBase.h>
#include <clang/Tooling/Tooling.h>

#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/MemoryBuffer.h>

#include <vast/Dialect/VastDialect.hpp>

#include <iostream>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <fstream>

namespace vast
{

    struct ASTVisitor : clang::RecursiveASTVisitor<ASTVisitor>
    {
        ASTVisitor(mlir::ModuleOp module)
            : module(module)
        {}

        virtual ~ASTVisitor() = default;

        virtual bool VisitDecl(clang::Decl *decl)
        {
            return decl->dump(), true;
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
} // namespace vast