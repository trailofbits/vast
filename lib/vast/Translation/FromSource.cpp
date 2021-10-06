// Copyright (c) 2021-present, Trail of Bits, Inc.

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <mlir/IR/Location.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Translation.h>
#include <mlir/IR/Builders.h>
#include <mlir/Dialect/SCF/SCF.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/IR/Identifier.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>

#include <clang/AST/RecursiveASTVisitor.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Frontend/ASTConsumers.h>
#include <clang/AST/Decl.h>
#include <clang/AST/Stmt.h>
#include <clang/AST/Expr.h>
#include <clang/AST/Type.h>
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
VAST_UNRELAX_WARNINGS

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
#include <optional>

#define DEBUG_TYPE "vast-from-source"

namespace vast::hl
{
    using string_ref = llvm::StringRef;
    using logical_result = mlir::LogicalResult;

    void spliceTrailingScopeBlocks(mlir::Region::BlockListType &blocks)
    {
        auto has_trailing_scope = [&] {
            if (blocks.empty())
                return false;
            auto &last_block = blocks.back();
            if (last_block.empty())
                return false;
            return mlir::isa< ScopeOp >(last_block.back());
        };

        while (has_trailing_scope()) {
            auto &last_block = blocks.back();

            auto scope  = mlir::cast< ScopeOp >(last_block.back());
            auto parent = scope.body().getParentRegion();
            scope->remove();

            auto &prev = parent->getBlocks().back();

            mlir::BlockAndValueMapping mapping;
            scope.body().cloneInto(parent, mapping);

            auto next = prev.getNextNode();

            auto &ops = last_block.getOperations();
            ops.splice(ops.end(), next->getOperations());

            next->erase();
            scope.erase();
        }
    }

    void spliceTrailingScopeBlocks(mlir::FuncOp &fn)
    {
        if (fn.empty())
            return;
        spliceTrailingScopeBlocks(fn.getBlocks());
    }

    void spliceTrailingScopeBlocks(mlir::Region &reg)
    {
        spliceTrailingScopeBlocks(reg.getBlocks());
    }

    struct VastBuilder
    {
        using OpBuilder = mlir::OpBuilder;
        using InsertPoint = OpBuilder::InsertPoint;

        VastBuilder(mlir::MLIRContext &mctx, mlir::OwningModuleRef &mod, clang::ASTContext &actx)
            : mctx(mctx), actx(actx), builder(mod->getBodyRegion())
        {}

        mlir::Location getLocation(clang::SourceRange range)
        {
            return getLocationImpl(range.getBegin());
        }

        mlir::Location getEndLocation(clang::SourceRange range)
        {
            return getLocationImpl(range.getEnd());
        }

        mlir::Location getLocationImpl(clang::SourceLocation at)
        {
            auto loc = actx.getSourceManager().getPresumedLoc(at);

            if (loc.isInvalid())
                return builder.getUnknownLoc();

            auto file = mlir::Identifier::get(loc.getFilename(), &mctx);
            return builder.getFileLineColLoc(file, loc.getLine(), loc.getColumn());

        }

        Value create_void(mlir::Location loc)
        {
            return builder.create< VoidOp >(loc);
        }

        template< typename Op, typename ...Args >
        auto create(Args &&... args)
        {
            return builder.create< Op >( std::forward< Args >(args)... );
        }

        InsertPoint saveInsertionPoint() { return builder.saveInsertionPoint(); }
        void restoreInsertionPoint(InsertPoint ip) { builder.restoreInsertionPoint(ip); }

        void setInsertionPointToStart(mlir::Block *block)
        {
            builder.setInsertionPointToStart(block);
        }

        void setInsertionPointToEnd(mlir::Block *block)
        {
            builder.setInsertionPointToEnd(block);
        }

        mlir::Block * getBlock() const { return builder.getBlock(); }

        mlir::Block * createBlock(mlir::Region *parent) { return builder.createBlock(parent); }

        mlir::Attribute constant_attr(const IntegerType &ty, int64_t value)
        {
            // TODO(Heno): make datalayout aware
            switch(ty.getKind()) {
                case vast::hl::integer_kind::Char:      return builder.getI8IntegerAttr(  char(value) );
                case vast::hl::integer_kind::Short:     return builder.getI16IntegerAttr( short(value) );
                case vast::hl::integer_kind::Int:       return builder.getI32IntegerAttr( int(value) );
                case vast::hl::integer_kind::Long:      return builder.getI64IntegerAttr( long(value) );
                case vast::hl::integer_kind::LongLong:  return builder.getI64IntegerAttr(value);
            }
        }

        mlir::Value constant(mlir::Location loc, mlir::Type ty, int64_t value)
        {
            if (ty.isa< IntegerType >()) {
                auto ity = ty.cast< IntegerType >();
                auto attr = constant_attr(ity, value);
                return builder.create< ConstantOp >(loc, ity, attr);
            }

            if (ty.isa< BoolType >()) {
                auto attr = builder.getBoolAttr(value);
                return builder.create< ConstantOp >(loc, ty, attr);
            }

            llvm_unreachable( "unsupported constant type" );
        }

    private:
        mlir::MLIRContext &mctx;
        clang::ASTContext &actx;

        OpBuilder builder;
    };


    struct ScopedInsertPoint
    {
        using InsertPoint = VastBuilder::InsertPoint;

        ScopedInsertPoint(VastBuilder &builder)
            : builder(builder), point(builder.saveInsertionPoint())
        {}

        ~ScopedInsertPoint()
        {
            builder.restoreInsertionPoint(point);
        }

        VastBuilder &builder;
        InsertPoint point;
    };

    struct VastDeclVisitor;

    struct VastStmtVisitor : clang::StmtVisitor< VastStmtVisitor, Value >
    {
        VastStmtVisitor(VastBuilder &builder, TypeConverter &types, VastDeclVisitor &decls)
            : builder(builder), types(types), decls(decls)
        {}

        void VisitFunction(clang::Stmt *body)
        {
            Visit(body);
        }

        Value VisitCompoundStmt(clang::CompoundStmt *stmt)
        {
            LLVM_DEBUG(llvm::dbgs() << "Visit CompoundStmt\n");

            ScopedInsertPoint builder_scope(builder);

            std::optional< ScopeOp > scope = std::nullopt;
            auto loc = builder.getLocation(stmt->getSourceRange());

            scope = builder.create< ScopeOp >(loc);
            auto &body = scope->body();
            body.push_back( new mlir::Block() );
            builder.setInsertionPointToStart( &body.front() );

            for (auto s : stmt->body()) {
                LLVM_DEBUG(llvm::dbgs() << "Visit Stmt " << s->getStmtClassName() << "\n");
                Visit(s);
            }

            if (scope.has_value()) {
                auto &body = scope->body();
                auto &lastblock = body.back();
                if (lastblock.empty() || lastblock.back().isKnownNonTerminator()) {
                    builder.setInsertionPointToEnd(&lastblock);
                    builder.create< ScopeEndOp >(loc);
                }
            }

            return Value(); // dummy return
        }

        auto make_scope_builder(clang::Stmt *stmt)
        {
            return [stmt, this] (auto &bld, auto) {
                Visit(stmt);
                spliceTrailingScopeBlocks(*bld.getBlock()->getParent());
            };
        }

        auto make_nonterminated_scope_builder(clang::Stmt *stmt)
        {
            return [stmt, this] (auto &bld, auto loc) {
                if (stmt)
                    Visit(stmt);
                spliceTrailingScopeBlocks(*bld.getBlock()->getParent());
                // TODO(Heno): remove with noterminator attribute
                auto &blocks = bld.getBlock()->getParent()->getBlocks();
                auto &lastblock = blocks.back();
                if (lastblock.empty() || lastblock.back().isKnownNonTerminator()) {
                    builder.setInsertionPointToEnd(&lastblock);
                    builder.create< ScopeEndOp >(loc);
                }
            };
        }

        Value VisitIfStmt(clang::IfStmt *stmt)
        {
            auto loc = builder.getLocation(stmt->getSourceRange());

            auto then_builder = make_scope_builder(stmt->getThen());

            auto cond = Visit(stmt->getCond());
            if (stmt->getElse()) {
                auto else_builder = make_scope_builder(stmt->getElse());
                builder.create< IfOp >(loc, cond, then_builder, else_builder);
            } else {
                builder.create< IfOp >(loc, cond, then_builder);
            }

            return Value(); // dummy return
        }

        Value VisitWhileStmt(clang::WhileStmt *stmt)
        {
            auto loc = builder.getLocation(stmt->getSourceRange());
            auto body_builder = make_scope_builder(stmt->getBody());

            auto cond = Visit(stmt->getCond());
            builder.create< WhileOp >(loc, cond, body_builder);

            return Value(); // dummy return
        }

        Value VisitForStmt(clang::ForStmt *stmt)
        {
            auto loc = builder.getLocation(stmt->getSourceRange());

            // auto incr = Visit(stmt->getInc());
            auto init_builder = make_nonterminated_scope_builder(stmt->getInit());
            auto cond_builder = make_nonterminated_scope_builder(stmt->getCond());
            auto inc_builder  = make_nonterminated_scope_builder(stmt->getInc());
            auto body_builder = make_scope_builder(stmt->getBody());

            builder.create< ForOp >(loc, init_builder, cond_builder, inc_builder, body_builder);

            return Value(); // dummy value
        }

        Value VisitReturnStmt(clang::ReturnStmt *stmt)
        {
            LLVM_DEBUG(llvm::dbgs() << "Visit ReturnStmt\n");
            auto loc = builder.getLocation(stmt->getSourceRange());
            if (stmt->getRetValue()) {
                auto val = Visit(stmt->getRetValue());

                // TODO(Heno): cast return values
                builder.create< ReturnOp >(loc, val);
            } else {
                auto val = builder.create_void(loc);
                builder.create< ReturnOp >(loc, val);
            }

            return Value(); // dummy value
        }

        Value VisitDeclStmt(clang::DeclStmt *stmt);

        Value VisitIntegerLiteral(const clang::IntegerLiteral *lit)
        {
            LLVM_DEBUG(llvm::dbgs() << "Visit IntegerLiteral\n");
            auto val = lit->getValue().getSExtValue();
            auto type = types.convert(lit->getType());
            auto loc = builder.getLocation(lit->getSourceRange());
            return builder.constant(loc, type, val);
        }

        Value VisitCXXBoolLiteralExpr(const clang::CXXBoolLiteralExpr *lit)
        {
            LLVM_DEBUG(llvm::dbgs() << "Visit CXXBoolLiteralExpr\n");
            bool val = lit->getValue();
            auto type = types.convert(lit->getType());
            auto loc = builder.getLocation(lit->getSourceRange());
            return builder.constant(loc, type, val);
        }

        Predicate integer_prdicate(clang::BinaryOperator *expr)
        {
            assert(expr->isComparisonOp());
            switch (expr->getOpcode()) {
                case clang::BinaryOperatorKind::BO_LT: return Predicate::slt;
                case clang::BinaryOperatorKind::BO_LE: return Predicate::sle;
                case clang::BinaryOperatorKind::BO_GT: return Predicate::sgt;
                case clang::BinaryOperatorKind::BO_GE: return Predicate::sge;
                case clang::BinaryOperatorKind::BO_EQ: return Predicate::eq;
                case clang::BinaryOperatorKind::BO_NE: return Predicate::ne;
                default: llvm_unreachable("unknown integer predicate");
            }
        }

        Value VisitComparisonOperator(clang::BinaryOperator *expr)
        {
            LLVM_DEBUG(llvm::dbgs() << "Visit ComparisonOperator\n");
            assert(expr->isComparisonOp());

            auto lhs = Visit(expr->getLHS());
            auto rhs = Visit(expr->getRHS());
            auto loc = builder.getEndLocation(expr->getSourceRange());

            // TODO(Heno): deal with floating point operations

            // TODO(Heno): zero extend arguments

            assert(lhs.getType() == rhs.getType());

            // TODO(Heno): deal with pointer comparisons
            auto pred = integer_prdicate(expr);
            return builder.create< CmpOp >(loc, pred, lhs, rhs);
        }

        Value VisitCompoundAssignOperator(clang::CompoundAssignOperator *expr)
        {
            LLVM_DEBUG(llvm::dbgs() << "Visit CompoundAssignOperator\n");

            auto lhs = Visit(expr->getLHS());
            auto rhs = Visit(expr->getRHS());
            auto loc = builder.getEndLocation(expr->getSourceRange());

            auto ty = expr->getType();
            switch (expr->getOpcode()) {
                case clang::BinaryOperatorKind::BO_AddAssign: {
                    if (ty->isIntegerType()) {
                        builder.create< AddIAssignOp >( loc, rhs, lhs );
                        break;
                    }
                    llvm_unreachable( "unhandled addition assign operation" );
                }
                case clang::BinaryOperatorKind::BO_SubAssign: {
                    if (ty->isIntegerType()) {
                        builder.create< SubIAssignOp >( loc, rhs, lhs );
                        break;
                    }
                    llvm_unreachable( "unhandled subtraction assign operation" );
                }
                case clang::BinaryOperatorKind::BO_MulAssign: {
                    if (ty->isIntegerType()) {
                        builder.create< MulIAssignOp >( loc, rhs, lhs );
                        break;
                    }
                    llvm_unreachable( "unhandled multiplication assign operation" );
                }
                case clang::BinaryOperatorKind::BO_DivAssign: {
                    if (ty->isUnsignedIntegerType()) {
                        builder.create< DivUAssignOp >( loc, rhs, lhs );
                        break;
                    }
                    if (ty->isIntegerType()) {
                        builder.create< DivSAssignOp >( loc, rhs, lhs );
                        break;
                    }
                    llvm_unreachable( "unhandled division assign operation" );
                }
                case clang::BinaryOperatorKind::BO_RemAssign: {
                    if (ty->isUnsignedIntegerType()) {
                        builder.create< RemUAssignOp >( loc, rhs, lhs );
                        break;
                    }
                    if (ty->isIntegerType()) {
                        builder.create< RemSAssignOp >( loc, rhs, lhs );
                        break;
                    }
                    llvm_unreachable( "unhandled reminder assign operation" );
                }
                default: {
                    llvm_unreachable( "unhandled compound assign operation" );
                }
            }

            return Value(); // dummy value
        }

        Value VisitBinaryOperator(clang::BinaryOperator *expr)
        {
            LLVM_DEBUG(llvm::dbgs() << "Visit BinaryOperator\n");

            if (expr->isComparisonOp()) {
                return VisitComparisonOperator(expr);
            }

            auto lhs = Visit(expr->getLHS());
            auto rhs = Visit(expr->getRHS());
            auto loc = builder.getEndLocation(expr->getSourceRange());

            auto ty = expr->getType();
            switch (expr->getOpcode()) {
                case clang::BinaryOperatorKind::BO_Add: {
                    if (ty->isIntegerType())
                        return builder.create< AddIOp >( loc, lhs, rhs );
                    llvm_unreachable( "unhandled addition type" );
                }
                case clang::BinaryOperatorKind::BO_Sub: {
                    if (ty->isIntegerType())
                        return builder.create< SubIOp >( loc, lhs, rhs );
                    llvm_unreachable( "unhandled subtraction type" );
                }
                case clang::BinaryOperatorKind::BO_Mul: {
                    if (ty->isIntegerType())
                        return builder.create< MulIOp >( loc, lhs, rhs );
                    llvm_unreachable( "unhandled multiplication type" );
                }
                case clang::BinaryOperatorKind::BO_Div: {
                    if (ty->isUnsignedIntegerType())
                        return builder.create< DivUOp >( loc, lhs, rhs );
                    if (ty->isIntegerType())
                        return builder.create< DivSOp >( loc, lhs, rhs );
                    llvm_unreachable( "unhandled division type" );
                }
                case clang::BinaryOperatorKind::BO_Rem: {
                    if (ty->isUnsignedIntegerType())
                        return builder.create< RemUOp >( loc, lhs, rhs );
                    if (ty->isIntegerType())
                        return builder.create< RemSOp >( loc, lhs, rhs );
                    llvm_unreachable( "unhandled reminder type" );
                }
                case clang::BinaryOperatorKind::BO_Assign: {
                    builder.create< AssignOp >( loc, rhs, lhs );
                    return Value();
                }
                default: {
                    llvm_unreachable( "unhandled binary operation" );
                }
            }

            return Value();
        }

        Value VisitUnaryOperator(clang::UnaryOperator *expr)
        {
            LLVM_DEBUG(llvm::dbgs() << "Visit UnaryOperator\n");
            auto loc = builder.getEndLocation(expr->getSourceRange());
            auto arg =  Visit(expr->getSubExpr());

            switch (expr->getOpcode()) {
                case clang::UnaryOperatorKind::UO_PostInc:
                    builder.create< PostIncOp >( loc, arg ); break;
                case clang::UnaryOperatorKind::UO_PostDec:
                    builder.create< PostDecOp >( loc, arg ); break;
                case clang::UnaryOperatorKind::UO_PreInc:
                    builder.create< PreIncOp >( loc, arg ); break;
                case clang::UnaryOperatorKind::UO_PreDec:
                    builder.create< PreDecOp >( loc, arg ); break;
                case clang::UnaryOperatorKind::UO_AddrOf:
                    llvm_unreachable( "unsupported unary address of operator" );
                case clang::UnaryOperatorKind::UO_Deref:
                    llvm_unreachable( "unsupported unary dereference operator" );
                case clang::UnaryOperatorKind::UO_Plus:
                    return builder.create< PlusOp >( loc, arg );
                case clang::UnaryOperatorKind::UO_Minus:
                    return builder.create< MinusOp >( loc, arg );
                case clang::UnaryOperatorKind::UO_Not:
                    return builder.create< NotOp >( loc, arg );
                case clang::UnaryOperatorKind::UO_LNot:
                    return builder.create< LNotOp >( loc, arg );
                case clang::UnaryOperatorKind::UO_Real:
                    llvm_unreachable( "unsupported `real` operator" );
                case clang::UnaryOperatorKind::UO_Imag:
                    llvm_unreachable( "unsupported `imag` operator" );
                case clang::UnaryOperatorKind::UO_Coawait:
                    llvm_unreachable( "unsupported `coawait` operator" );
                case clang::UnaryOperatorKind::UO_Extension:
                    llvm_unreachable( "unsupported `extension` operator" );
            }

            return Value(); // dummy value
        }

        Value VisitImplicitCastExpr(clang::ImplicitCastExpr *expr)
        {
            LLVM_DEBUG(llvm::dbgs() << "Visit ImplicitCastExpr\n");
            auto loc = builder.getLocation(expr->getSourceRange());
            auto value = Visit(expr->getSubExpr());
            auto rty = types.convert(expr->getType());
            return builder.create< ImplicitCastOp >( loc, rty, value );
        }

        Value VisitDeclRefExpr(clang::DeclRefExpr *decl)
        {
            LLVM_DEBUG(llvm::dbgs() << "Visit DeclRefExpr\n");
            auto loc = builder.getLocation(decl->getSourceRange());

            // TODO(Heno): deal with function declaration

            // TODO(Heno): deal with enum constant declaration

            auto named = decl->getDecl()->getUnderlyingDecl();
            auto rty = types.convert(decl->getType());
            return builder.create< DeclRefOp >( loc, rty, named->getNameAsString() );

        }

    private:

        VastBuilder     &builder;
        TypeConverter   &types;
        VastDeclVisitor &decls;
    };

    struct VastDeclVisitor : clang::DeclVisitor< VastDeclVisitor, mlir::Value >
    {
        VastDeclVisitor(mlir::MLIRContext &mctx, mlir::OwningModuleRef &mod, clang::ASTContext &actx)
            : mod(mod)
            , builder(mctx, mod, actx)
            , types(&mctx)
            , stmts(builder, types, *this)
        {}

        mlir::Location getLocation(clang::SourceRange range)
        {
            return builder.getLocation(range);
        }

        mlir::Location getEndLocation(clang::SourceRange range)
        {
            return builder.getEndLocation(range);
        }

        template< typename T >
        decltype(auto) convert(T type) { return types.convert(type); }

        Value VisitFunctionDecl(clang::FunctionDecl *decl)
        {
            LLVM_DEBUG(llvm::dbgs() << "Visit FunctionDecl: " << decl->getName() << "\n");
            ScopedInsertPoint builder_scope(builder);
            llvm::ScopedHashTableScope scope(symbols);

            auto loc  = getLocation(decl->getSourceRange());
            auto type = convert(decl->getFunctionType());
            assert( type );

            auto fn = builder.create< mlir::FuncOp >(loc, decl->getName(), type);

            // TODO(Heno): move to function prototype lifting
            if (!decl->isMain())
                fn.setVisibility(mlir::FuncOp::Visibility::Private);

            if (!decl->hasBody() || !fn.isExternal())
                return Value(); // dummy value

            auto entry = fn.addEntryBlock();

            // In MLIR the entry block of the function must have the same argument list as the function itself.
            for (const auto &[arg, earg] : llvm::zip(decl->parameters(), entry->getArguments())) {
                if (failed(declare(arg->getName(), earg)))
                    mod->emitError("multiple declarations of a same symbol" + arg->getName());
            }

            builder.setInsertionPointToStart(entry);

            // emit function body
            if (decl->hasBody()) {
                stmts.VisitFunction(decl->getBody());
            }

            spliceTrailingScopeBlocks(fn);

            auto &last_block = fn.getBlocks().back();
            auto &ops = last_block.getOperations();
            builder.setInsertionPointToEnd(&last_block);

            if (ops.empty() || ops.back().isKnownNonTerminator()) {
                auto beg_loc = getLocation(decl->getBeginLoc());
                auto end_loc = getLocation(decl->getEndLoc());
                if (decl->getReturnType()->isVoidType()) {
                    auto val = builder.create_void(end_loc);
                    builder.create< ReturnOp >(end_loc, val);
                } else {
                    if (decl->isMain()) {
                        // return zero if no return is present in main
                        auto zero = builder.constant(end_loc, type.getResult(0), 0);
                        builder.create< ReturnOp >(end_loc, zero);
                    } else {
                        builder.create< UnreachableOp >(beg_loc);
                    }
                }
            }

            return Value(); // dummy value
        }

        Value VisitVarDecl(clang::VarDecl *decl)
        {
            LLVM_DEBUG(llvm::dbgs() << "Visit VarDecl\n");
            auto ty    = convert(decl->getType());
            auto named = decl->getUnderlyingDecl();
            auto loc   = getEndLocation(decl->getSourceRange());
            auto init  = decl->getInit();

            if (init) {
                auto initializer = stmts.Visit(init);
                return builder.create< VarOp >(loc, ty, named->getName(), initializer);
            } else {
                return builder.create< VarOp >(loc, ty, named->getName());
            }
        }

    private:
        mlir::OwningModuleRef &mod;

        VastBuilder builder;
        TypeConverter types;

        VastStmtVisitor stmts;

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

    Value VastStmtVisitor::VisitDeclStmt(clang::DeclStmt *stmt)
    {
        LLVM_DEBUG(llvm::dbgs() << "Visit DeclStmt\n");
        for (auto decl : stmt->decls()) {
            LLVM_DEBUG(llvm::dbgs() << "Visit Decl "  << decl->getDeclKindName() << "\n");
            decls.Visit(decl);
        }

        return Value(); // dummy value
    }

    struct VastCodeGen : clang::ASTConsumer
    {
        VastCodeGen(mlir::MLIRContext &mctx, mlir::OwningModuleRef &mod, clang::ASTContext &actx)
            : mctx(mctx), mod(mod), actx(actx)
        {}

        bool HandleTopLevelDecl(clang::DeclGroupRef) override
        {
            llvm_unreachable("not implemented");
        }

        void HandleTranslationUnit(clang::ASTContext&) override
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
        ctx->loadDialect< mlir::scf::SCFDialect >();

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
