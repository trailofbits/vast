// Copyright (c) 2021-present, Trail of Bits, Inc.

#include <clang/AST/ASTConsumer.h>
#include <clang/AST/DeclBase.h>
#include <clang/Tooling/Tooling.h>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Translation.h>
#include <mlir/Support/LogicalResult.h>

#include <clang/AST/RecursiveASTVisitor.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Frontend/ASTConsumers.h>
#include <clang/Frontend/FrontendActions.h>

#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/MemoryBuffer.h>

#include <iostream>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>

namespace vast
{

    struct ASTVisitor : clang::RecursiveASTVisitor<ASTVisitor>
    {
        virtual ~ASTVisitor() = default;

        virtual bool VisitDecl(clang::Decl *decl)
        {
            return decl->dump(), true;
        }
    };

    struct ASTConsumer : clang::ASTConsumer
    {
        virtual void HandleTranslationUnit(clang::ASTContext &ctx) override
        {
            visitor.TraverseDecl(ctx.getTranslationUnitDecl());
        }

    private:
        ASTVisitor visitor;
    };

    struct ASTAction : clang::ASTFrontendAction
    {
        using Compiler = clang::CompilerInstance;
        using Consumer = std::unique_ptr< clang::ASTConsumer >;

        virtual Consumer CreateASTConsumer(Compiler &cc, llvm::StringRef in) override
        {
            return std::make_unique< ASTConsumer >();
        }
    };

    static mlir::OwningModuleRef from_source_parser(const llvm::MemoryBuffer *input, mlir::MLIRContext *ctx)
    {
        clang::tooling::runToolOnCode(std::make_unique<ASTAction>(), input->getBuffer());

        return {};
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