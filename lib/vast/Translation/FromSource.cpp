// Copyright (c) 2021-present, Trail of Bits, Inc.

#include <mlir/IR/Location.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Translation.h>
#include <mlir/IR/Builders.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/Identifier.h>

#include <clang/AST/RecursiveASTVisitor.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Frontend/ASTConsumers.h>
#include <clang/AST/Decl.h>
#include <clang/AST/Stmt.h>
#include <clang/Frontend/FrontendAction.h>
#include <clang/AST/ASTConsumer.h>
#include <clang/AST/DeclBase.h>
#include <clang/Tooling/Tooling.h>
#include <clang/AST/ASTContext.h>
#include <clang/AST/DeclVisitor.h>


#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/ADT/ScopedHashTable.h>
#include <llvm/Support/Debug.h>
#include <llvm/ADT/None.h>
#include <llvm/Support/ErrorHandling.h>

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Value.h"
#include "vast/Translation/Types.hpp"
#include "vast/Dialect/HighLevel/HighLevel.hpp"
#include "vast/Dialect/HighLevel/HighLevelTypes.hpp"

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
    using string_ref = llvm::StringRef;
    using logical_result = mlir::LogicalResult;

    struct VastBuilder
    {
        using OpBuilder = mlir::OpBuilder;

        VastBuilder(mlir::MLIRContext &mctx, mlir::OwningModuleRef &mod, clang::ASTContext &actx)
            : mctx(mctx), actx(actx), builder(mod->getBodyRegion())
        {}

        mlir::Location getLocation(clang::SourceRange range)
        {
            auto beg = range.getBegin();
            auto loc = actx.getSourceManager().getPresumedLoc(beg);

            if (loc.isInvalid())
                return builder.getUnknownLoc();

            auto file = mlir::Identifier::get(loc.getFilename(), &mctx);
            return builder.getFileLineColLoc(file, loc.getLine(), loc.getColumn());
        }

        template< typename Op, typename ...Args >
        auto create(Args &&... args)
        {
            return builder.create< Op >( std::forward< Args >(args)... );
        }

        void setInsertionPointToStart(mlir::Block *block)
        {
            builder.setInsertionPointToStart(block);
        }

        mlir::Block * getBlock() const { return builder.getBlock(); }

        mlir::Attribute constant_attr(const IntegerType &ty, int64_t value)
        {
            // TODO(Heno): make datalayout aware
            switch(ty.getKind()) {
                case vast::hl::integer_kind::Char:      return builder.getI8IntegerAttr(value);
                case vast::hl::integer_kind::Short:     return builder.getI16IntegerAttr(value);
                case vast::hl::integer_kind::Int:       return builder.getI32IntegerAttr(value);
                case vast::hl::integer_kind::Long:      return builder.getI64IntegerAttr(value);
                case vast::hl::integer_kind::LongLong:  return builder.getI64IntegerAttr(value);
            }
        }

        mlir::Value constant(mlir::Location loc, mlir::Type ty, int64_t value)
        {
            assert(ty.isa< IntegerType >());
            auto ity = ty.cast< IntegerType >();
            auto attr = constant_attr(ity, value);
            return builder.create< ConstantOp >(loc, ity, attr);
        }

    private:
        mlir::MLIRContext &mctx;
        clang::ASTContext &actx;

        OpBuilder builder;
    };

    struct VastExprVisitor : clang::StmtVisitor< VastExprVisitor, mlir::Value >
    {
        VastExprVisitor(VastBuilder &builder, TypeConverter &types)
            : builder(builder), types(types)
        {}

        mlir::Value VisitIntegerLiteral(const clang::IntegerLiteral *lit)
        {
            LLVM_DEBUG(llvm::dbgs() << "Visit IntegerLiteral\n");
            auto val = lit->getValue().getSExtValue();
            auto type = types.convert(lit->getType());
            auto loc = builder.getLocation(lit->getSourceRange());
            return builder.constant(loc, type, val);
        }

    private:

        VastBuilder &builder;
        TypeConverter &types;
    };

    struct VastStmtVisitor : clang::StmtVisitor< VastStmtVisitor >
    {
        VastStmtVisitor(VastBuilder &builder, TypeConverter &types)
            : builder(builder), types(types)
        {}

        void VisitCompoundStmt(clang::CompoundStmt *stmt)
        {
            LLVM_DEBUG(llvm::dbgs() << "Visit CompoundStmt\n");

            for (auto s : stmt->body()) {
                LLVM_DEBUG(llvm::dbgs() << "Visit Stmt " << s->getStmtClassName() << "\n");
                Visit(s);
            }
        }

        void VisitReturnStmt(clang::ReturnStmt *stmt)
        {
            LLVM_DEBUG(llvm::dbgs() << "Visit ReturnStmt\n");
            auto loc = builder.getLocation(stmt->getSourceRange());

            if (stmt->getRetValue()) {
                VastExprVisitor visitor(builder, types);
                auto val = visitor.Visit(stmt->getRetValue());

                // TODO(Heno): cast return values
                builder.create< mlir::ReturnOp >(loc, val);
            } else {
                builder.create< mlir::ReturnOp >(loc);
            }
        }

    private:

        VastBuilder &builder;
        TypeConverter &types;
    };

    struct VastDeclVisitor : clang::DeclVisitor< VastDeclVisitor >
    {

        VastDeclVisitor(mlir::MLIRContext &mctx, mlir::OwningModuleRef &mod, clang::ASTContext &actx)
            : mctx(mctx), mod(mod),  actx(actx), builder(mctx, mod, actx), types(&mctx)
        {}

        void VisitFunctionDecl(clang::FunctionDecl *decl)
        {
            LLVM_DEBUG(llvm::dbgs() << "Visit FunctionDecl: " << decl->getName() << "\n");
            llvm::ScopedHashTableScope scope(symbols);

            auto loc = builder.getLocation(decl->getSourceRange());
            auto type = types.convert(decl->getFunctionType());
            assert( type );

            auto fn = builder.create< mlir::FuncOp >(loc, decl->getName(), type);

            // TODO(Heno): move to function prototype lifting
            if (!decl->isMain())
                fn.setVisibility(mlir::FuncOp::Visibility::Private);

            if (!decl->hasBody() || !fn.isExternal())
                return;

            auto entry = fn.addEntryBlock();

            // In MLIR the entry block of the function must have the same argument list as the function itself.
            for (const auto &[arg, earg] : llvm::zip(decl->parameters(), entry->getArguments())) {
                if (failed(declare(arg->getName(), earg)))
                    mod->emitError("multiple declarations of a same symbol" + arg->getName());
            }

            builder.setInsertionPointToStart(entry);

            if (decl->hasBody()) {
                VastStmtVisitor visitor(builder, types);
                visitor.Visit(decl->getBody());
            }

            auto end = builder.getBlock();
            auto &ops = end->getOperations();
            if (ops.empty() || ops.back().isKnownNonTerminator()) {
                if (decl->getReturnType()->isVoidType()) {
                    builder.create< mlir::ReturnOp >(builder.getLocation(decl->getEndLoc()));
                } else {
                    builder.create< UnreachableOp >(builder.getLocation(decl->getBeginLoc()));
                }
            }
        }

    private:
        mlir::MLIRContext     &mctx;
        mlir::OwningModuleRef &mod;
        clang::ASTContext     &actx;

        VastBuilder builder;
        TypeConverter types;

        // Declare a variable in the current scope, return success if the variable
        // wasn't declared yet.
        logical_result declare(string_ref var, mlir::Value value)
        {
            if (symbols.count(var))
                return mlir::failure();
            symbols.insert(var, value);
            return mlir::success();
        }

        // The symbol table maps a variable name to a value in the current scope.
        // Entering a function creates a new scope, and the function arguments are
        // added to the mapping. When the processing of a function is terminated, the
        // scope is destroyed and the mappings created in this scope are dropped.
        llvm::ScopedHashTable<string_ref, mlir::Value> symbols;
    };

    struct VastCodeGen : clang::ASTConsumer
    {
        VastCodeGen(mlir::MLIRContext &mctx, mlir::OwningModuleRef &mod, clang::ASTContext &actx)
            : mctx(mctx), mod(mod), actx(actx)
        {}

        bool HandleTopLevelDecl(clang::DeclGroupRef decls) override
        {
            llvm_unreachable("not implemented");
        }

        void HandleTranslationUnit(clang::ASTContext &ctx) override
        {
            LLVM_DEBUG(llvm::dbgs() << "Process Translation Unit\n");
            auto tu = actx.getTranslationUnitDecl();

            VastDeclVisitor visitor(mctx, mod, actx);

            for (const auto &decl : tu->decls())
                visitor.Visit(decl);
        }

    private:
        mlir::MLIRContext     &mctx;
        mlir::OwningModuleRef &mod;
        clang::ASTContext     &actx;
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

        VastCodeGen codegen(*ctx, module, ast->getASTContext());
        codegen.HandleTranslationUnit(ast->getASTContext());

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
