// Copyright (c) 2021-present, Trail of Bits, Inc.

#include "vast/Util/Warnings.hpp"
#include "vast/Util/Functions.hpp"

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
#include "vast/Translation/Expr.hpp"
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
#include <variant>

#define DEBUG_TYPE "vast-from-source"

namespace vast::hl
{
    using string_ref = llvm::StringRef;
    using logical_result = mlir::LogicalResult;

    using Stmt = mlir::Operation*;
    using ValueOrStmt = std::variant< mlir::Value, Stmt >;

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

        static inline auto to_value = [] (ValueOrStmt v) -> Value
        {
            return std::get< Value >(v);
        };

        static inline auto convert = overloaded{ to_value, identity };

        template< typename Op, typename ...Args >
        auto create(Args &&... args)
        {
            return builder.create< Op >( convert( std::forward< Args >(args) )... );
        }

        template< typename Op, typename ...Args >
        Value create_value(Args &&... args)
        {
            return create< Op >( std::forward< Args >(args)... );
        }

        template< typename Op, typename ...Args >
        Stmt create_stmt(Args &&... args)
        {
            return create< Op >( std::forward< Args >(args)... );
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

    struct VastCodeGenVisitor : clang::StmtVisitor< VastCodeGenVisitor, ValueOrStmt >
    {
        VastCodeGenVisitor(VastBuilder &builder, TypeConverter &types, VastDeclVisitor &decls)
            : builder(builder), types(types), decls(decls)
        {}

        void VisitFunction(clang::Stmt *body)
        {
            Visit(body);
        }

        template< typename Op >
        ValueOrStmt build_binary(clang::BinaryOperator *expr)
        {
            auto lhs = Visit(expr->getLHS());
            auto rhs = Visit(expr->getRHS());
            auto loc = builder.getEndLocation(expr->getSourceRange());
            auto res = builder.create< Op >( loc, lhs, rhs );

            if constexpr ( std::is_convertible_v< decltype(res), Value > ) {
                return Value(res);
            } else {
                return res;
            }
        }

        template< Predicate pred >
        ValueOrStmt build_comparison(clang::BinaryOperator *expr)
        {
            auto lhs = Visit(expr->getLHS());
            auto rhs = Visit(expr->getRHS());
            auto loc = builder.getEndLocation(expr->getSourceRange());
            return builder.create_value< CmpOp >( loc, pred, lhs, rhs );
        }

        template< typename Op >
        ValueOrStmt build_unary(clang::UnaryOperator *expr)
        {
            auto loc = builder.getEndLocation(expr->getSourceRange());
            auto arg = Visit(expr->getSubExpr());
            auto res = builder.create< Op >( loc, arg );
            if constexpr ( std::is_convertible_v< decltype(res), Value > ) {
                return Value(res);
            } else {
                return res;
            }
        }

        template< typename Cast >
        ValueOrStmt build_cast(clang::Expr *expr, clang::QualType to, CastKind kind)
        {
            auto loc = builder.getLocation(expr->getSourceRange());
            auto rty = types.convert(to);
            return builder.create_value< Cast >( loc, rty, Visit(expr), kind );
        }

        ValueOrStmt VisitBinPtrMemD(clang::BinaryOperator *expr)
        {
            llvm_unreachable( "unhandled BinPtrMemD" );
        }

        ValueOrStmt VisitBinPtrMemI(clang::BinaryOperator *expr)
        {
            llvm_unreachable( "unhandled BinPtrMemI" );
        }

        ValueOrStmt VisitBinMul(clang::BinaryOperator *expr)
        {
            auto ty = expr->getType();
            if (ty->isIntegerType())
                return build_binary< MulIOp >(expr);
            llvm_unreachable( "unhandled BinMul" );
        }

        ValueOrStmt VisitBinDiv(clang::BinaryOperator *expr)
        {
            auto ty = expr->getType();
            if (ty->isUnsignedIntegerType())
                return build_binary< DivUOp >(expr);
            if (ty->isIntegerType())
                return build_binary< DivSOp >(expr);
            llvm_unreachable( "unhandled BinDiv" );
        }

        ValueOrStmt VisitBinRem(clang::BinaryOperator *expr)
        {
            auto ty = expr->getType();
            if (ty->isUnsignedIntegerType())
                return build_binary< RemUOp >(expr);
            if (ty->isIntegerType())
                return build_binary< RemSOp >(expr);
            llvm_unreachable( "unhandled BinRem" );
        }

        ValueOrStmt VisitBinAdd(clang::BinaryOperator *expr)
        {
            auto ty = expr->getType();
            if (ty->isIntegerType())
                return build_binary< AddIOp >(expr);
            llvm_unreachable( "unhandled addition type" );
        }

        ValueOrStmt VisitBinSub(clang::BinaryOperator *expr)
        {
            auto ty = expr->getType();
            if (ty->isIntegerType())
                return build_binary< SubIOp >(expr);
            llvm_unreachable( "unhandled BinSub" );
        }

        ValueOrStmt VisitBinShl(clang::BinaryOperator *expr)
        {
            llvm_unreachable( "unhandled BinShl" );
        }

        ValueOrStmt VisitBinShr(clang::BinaryOperator *expr)
        {
            llvm_unreachable( "unhandled BinShr" );
        }

        ValueOrStmt VisitBinLT(clang::BinaryOperator *expr)
        {
            auto ty = expr->getLHS()->getType();
            if (ty->isUnsignedIntegerType())
                return build_comparison< Predicate::ult >(expr);
            if (ty->isIntegerType())
                return build_comparison< Predicate::slt >(expr);
            llvm_unreachable( "unhandled BinLT" );
        }

        ValueOrStmt VisitBinGT(clang::BinaryOperator *expr)
        {
            auto ty = expr->getLHS()->getType();
            if (ty->isUnsignedIntegerType())
                return build_comparison< Predicate::ugt >(expr);
            if (ty->isIntegerType())
                return build_comparison< Predicate::sgt >(expr);
            llvm_unreachable( "unhandled BinGT" );
        }

        ValueOrStmt VisitBinLE(clang::BinaryOperator *expr)
        {
            auto ty = expr->getLHS()->getType();
            if (ty->isUnsignedIntegerType())
                return build_comparison< Predicate::ule >(expr);
            if (ty->isIntegerType())
                return build_comparison< Predicate::sle >(expr);
            llvm_unreachable( "unhandled BinLE" );
        }

        ValueOrStmt VisitBinGE(clang::BinaryOperator *expr)
        {
            auto ty = expr->getLHS()->getType();
            if (ty->isUnsignedIntegerType())
                return build_comparison< Predicate::uge >(expr);
            if (ty->isIntegerType())
                return build_comparison< Predicate::sge >(expr);
            llvm_unreachable( "unhandled BinGE" );
        }

        ValueOrStmt VisitBinEQ(clang::BinaryOperator *expr)
        {
            auto ty = expr->getLHS()->getType();
            if (ty->isIntegerType())
                return build_comparison< Predicate::eq >(expr);
            llvm_unreachable( "unhandled BinEQ" );
        }

        ValueOrStmt VisitBinNE(clang::BinaryOperator *expr)
        {
            auto ty = expr->getLHS()->getType();
            if (ty->isIntegerType())
                return build_comparison< Predicate::ne >(expr);
            llvm_unreachable( "unhandled BinNE" );
        }

        ValueOrStmt VisitBinAnd(clang::BinaryOperator *expr)
        {
            llvm_unreachable( "unhandled BinAnd" );
        }

        ValueOrStmt VisitBinXor(clang::BinaryOperator *expr)
        {
            llvm_unreachable( "unhandled BinXor" );
        }

        ValueOrStmt VisitBinOr(clang::BinaryOperator *expr)
        {
            llvm_unreachable( "unhandled BinOr" );
        }

        ValueOrStmt VisitBinLAnd(clang::BinaryOperator *expr)
        {
            llvm_unreachable( "unhandled BinLAnd" );
        }

        ValueOrStmt VisitBinLOr(clang::BinaryOperator *expr)
        {
            llvm_unreachable( "unhandled BinLOr" );
        }

        ValueOrStmt VisitAssign(clang::BinaryOperator *expr)
        {
            return build_binary< AssignOp >(expr);
        }

        ValueOrStmt VisitBinMulAssign(clang::CompoundAssignOperator *expr)
        {
            auto ty = expr->getType();
            if (ty->isIntegerType())
                return build_binary< MulIAssignOp >(expr);
            llvm_unreachable( "unhandled BinMulAssign" );
        }

        ValueOrStmt VisitBinDivAssign(clang::CompoundAssignOperator *expr)
        {
            auto ty = expr->getType();
            if (ty->isUnsignedIntegerType())
                return build_binary< DivUAssignOp >(expr);
            if (ty->isIntegerType())
                return build_binary< DivSAssignOp >(expr);
            llvm_unreachable( "unhandled BinDivAssign" );
        }

        ValueOrStmt VisitBinRemAssign(clang::CompoundAssignOperator *expr)
        {
            auto ty = expr->getType();
            if (ty->isUnsignedIntegerType())
                return build_binary< RemUAssignOp >(expr);
            if (ty->isIntegerType())
                return build_binary< RemSAssignOp >(expr);
            llvm_unreachable( "unhandled BinRemAssign" );
        }

        ValueOrStmt VisitBinAddAssign(clang::CompoundAssignOperator *expr)
        {
            auto ty = expr->getType();
            if (ty->isIntegerType())
                return build_binary< AddIAssignOp >(expr);
            llvm_unreachable( "unhandled BinAddAssign" );
        }

        ValueOrStmt VisitBinSubAssign(clang::CompoundAssignOperator *expr)
        {
            auto ty = expr->getType();
            if (ty->isIntegerType())
                return build_binary< SubIAssignOp >(expr);
            llvm_unreachable( "unhandled BinSubAssign" );
        }

        ValueOrStmt VisitBinShlAssign(clang::CompoundAssignOperator *expr)
        {
            llvm_unreachable( "unhandled BinShlAssign" );
        }

        ValueOrStmt VisitBinShrAssign(clang::CompoundAssignOperator *expr)
        {
            llvm_unreachable( "unhandled BinShrAssign" );
        }

        ValueOrStmt VisitBinAndAssign(clang::CompoundAssignOperator *expr)
        {
            llvm_unreachable( "unhandled BinAndAssign" );
        }

        ValueOrStmt VisitBinOrAssign(clang::CompoundAssignOperator *expr)
        {
            llvm_unreachable( "unhandled BinOrAssign" );
        }

        ValueOrStmt VisitBinXorAssign(clang::CompoundAssignOperator *expr)
        {
            llvm_unreachable( "unhandled BinXorAssign" );
        }

        ValueOrStmt VisitBinComma(clang::BinaryOperator *expr)
        {
            llvm_unreachable( "unhandled BinComma" );
        }

        ValueOrStmt VisitUnaryPostInc(clang::UnaryOperator *expr)
        {
            return build_unary< PostIncOp >(expr);
        }

        ValueOrStmt VisitUnaryPostDec(clang::UnaryOperator *expr)
        {
            return build_unary< PostDecOp >(expr);
        }

        ValueOrStmt VisitUnaryPreInc(clang::UnaryOperator *expr)
        {
            return build_unary< PreIncOp >(expr);
        }

        ValueOrStmt VisitUnaryPreDec(clang::UnaryOperator *expr)
        {
            return build_unary< PreDecOp >(expr);
        }

        ValueOrStmt VisitUnaryAddrOf(clang::UnaryOperator *expr)
        {
            llvm_unreachable( "unhandled UnaryAddrOf" );
        }

        ValueOrStmt VisitUnaryDeref(clang::UnaryOperator *expr)
        {
            llvm_unreachable( "unhandled UnaryDeref" );
        }

        ValueOrStmt VisitUnaryPlus(clang::UnaryOperator *expr)
        {
            return build_unary< PlusOp >(expr);
        }

        ValueOrStmt VisitUnaryMinus(clang::UnaryOperator *expr)
        {
            return build_unary< MinusOp >(expr);
        }

        ValueOrStmt VisitUnaryNot(clang::UnaryOperator *expr)
        {
            return build_unary< NotOp >(expr);
        }

        ValueOrStmt VisitUnaryLNot(clang::UnaryOperator *expr)
        {
            return build_unary< LNotOp >(expr);
        }

        ValueOrStmt VisitUnaryReal(clang::UnaryOperator *expr)
        {
            llvm_unreachable( "unhandled UnaryReal" );
        }

        ValueOrStmt VisitUnaryImag(clang::UnaryOperator *expr)
        {
            llvm_unreachable( "unhandled UnaryImag" );
        }

        ValueOrStmt VisitUnaryExtension(clang::UnaryOperator *expr)
        {
            llvm_unreachable( "unhandled UnaryExtension" );
        }

        ValueOrStmt VisitUnaryCoawait(clang::UnaryOperator *expr)
        {
            llvm_unreachable( "unhandled UnaryCoawait" );
        }

        ValueOrStmt VisitCompoundStmt(clang::CompoundStmt *stmt)
        {
            LLVM_DEBUG(llvm::dbgs() << "Visit CompoundStmt\n");

            ScopedInsertPoint builder_scope(builder);

            auto loc = builder.getLocation(stmt->getSourceRange());

            ScopeOp scope = builder.create< ScopeOp >(loc);
            auto &body = scope.body();
            body.push_back( new mlir::Block() );
            builder.setInsertionPointToStart( &body.front() );

            for (auto s : stmt->body()) {
                LLVM_DEBUG(llvm::dbgs() << "Visit Stmt " << s->getStmtClassName() << "\n");
                Visit(s);
            }

            auto &lastblock = body.back();
            if (lastblock.empty() || lastblock.back().isKnownNonTerminator()) {
                builder.setInsertionPointToEnd(&lastblock);
                builder.create< ScopeEndOp >(loc);
            }

            return scope;
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

        ValueOrStmt VisitIfStmt(clang::IfStmt *stmt)
        {
            auto loc = builder.getLocation(stmt->getSourceRange());

            auto then_builder = make_scope_builder(stmt->getThen());

            auto cond = Visit(stmt->getCond());
            if (stmt->getElse()) {
                auto else_builder = make_scope_builder(stmt->getElse());
                return builder.create< IfOp >(loc, cond, then_builder, else_builder);
            } else {
                return builder.create< IfOp >(loc, cond, then_builder);
            }
        }

        ValueOrStmt VisitWhileStmt(clang::WhileStmt *stmt)
        {
            auto loc = builder.getLocation(stmt->getSourceRange());
            auto body_builder = make_scope_builder(stmt->getBody());

            auto cond = Visit(stmt->getCond());
            return builder.create< WhileOp >(loc, cond, body_builder);
        }

        ValueOrStmt VisitForStmt(clang::ForStmt *stmt)
        {
            auto loc = builder.getLocation(stmt->getSourceRange());

            auto init_builder = make_nonterminated_scope_builder(stmt->getInit());
            auto cond_builder = make_nonterminated_scope_builder(stmt->getCond());
            auto incr_builder = make_nonterminated_scope_builder(stmt->getInc());
            auto body_builder = make_scope_builder(stmt->getBody());

            return builder.create< ForOp >(loc, init_builder, cond_builder, incr_builder, body_builder);
        }

        ValueOrStmt VisitReturnStmt(clang::ReturnStmt *stmt)
        {
            LLVM_DEBUG(llvm::dbgs() << "Visit ReturnStmt\n");
            auto loc = builder.getLocation(stmt->getSourceRange());
            if (stmt->getRetValue()) {
                auto val = Visit(stmt->getRetValue());
                return builder.create< ReturnOp >(loc, val);
            } else {
                auto val = builder.create_void(loc);
                return builder.create< ReturnOp >(loc, val);
            }
        }

        ValueOrStmt VisitDeclStmt(clang::DeclStmt *stmt);

        template< typename Value, typename Literal >
        ValueOrStmt VisitLiteral(Value val, Literal lit)
        {
            auto type = types.convert(lit->getType());
            auto loc = builder.getLocation(lit->getSourceRange());
            return builder.constant(loc, type, val);
        }

        ValueOrStmt VisitIntegerLiteral(const clang::IntegerLiteral *lit)
        {
            LLVM_DEBUG(llvm::dbgs() << "Visit IntegerLiteral\n");
            auto val = lit->getValue().getSExtValue();
            return VisitLiteral(val, lit);
        }

        ValueOrStmt VisitCXXBoolLiteralExpr(const clang::CXXBoolLiteralExpr *lit)
        {
            LLVM_DEBUG(llvm::dbgs() << "Visit CXXBoolLiteralExpr\n");
            bool val = lit->getValue();
            return VisitLiteral(val, lit);
        }

        ValueOrStmt VisitImplicitCastExpr(clang::ImplicitCastExpr *expr)
        {
            LLVM_DEBUG(llvm::dbgs() << "Visit ImplicitCastExpr\n");
            return build_cast< ImplicitCastOp >( expr->getSubExpr(), expr->getType(), cast_kind(expr) );
        }

        ValueOrStmt VisitCStyleCastExpr(clang::CStyleCastExpr *expr)
        {
            LLVM_DEBUG(llvm::dbgs() << "Visit CStyleCastExpr\n");
            return build_cast< CStyleCastOp >( expr->getSubExpr(), expr->getType(), cast_kind(expr) );
        }

        ValueOrStmt VisitBuiltinBitCastExpr(clang::BuiltinBitCastExpr *expr)
        {
            LLVM_DEBUG(llvm::dbgs() << "Visit BuiltinBitCastExpr\n");
            return build_cast< BuiltinBitCastOp >( expr->getSubExpr(), expr->getType(), cast_kind(expr) );
        }

        ValueOrStmt VisitDeclRefExpr(clang::DeclRefExpr *decl)
        {
            LLVM_DEBUG(llvm::dbgs() << "Visit DeclRefExpr\n");
            auto loc = builder.getLocation(decl->getSourceRange());

            // TODO(Heno): deal with function declaration

            // TODO(Heno): deal with enum constant declaration

            auto named = decl->getDecl()->getUnderlyingDecl();
            auto rty = types.convert(decl->getType());
            return builder.create_value< DeclRefOp >( loc, rty, named->getNameAsString() );
        }

    private:

        VastBuilder     &builder;
        TypeConverter   &types;
        VastDeclVisitor &decls;
    };

    struct VastDeclVisitor : clang::DeclVisitor< VastDeclVisitor, ValueOrStmt >
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

        ValueOrStmt VisitFunctionDecl(clang::FunctionDecl *decl)
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

            return fn;
        }

        ValueOrStmt VisitVarDecl(clang::VarDecl *decl)
        {
            LLVM_DEBUG(llvm::dbgs() << "Visit VarDecl\n");
            auto ty    = convert(decl->getType());
            auto named = decl->getUnderlyingDecl();
            auto loc   = getEndLocation(decl->getSourceRange());
            auto init  = decl->getInit();

            if (init) {
                auto initializer = stmts.Visit(init);
                return builder.create_value< VarOp >(loc, ty, named->getName(), initializer);
            } else {
                return builder.create_value< VarOp >(loc, ty, named->getName());
            }
        }

    private:
        mlir::OwningModuleRef &mod;

        VastBuilder builder;
        TypeConverter types;

        VastCodeGenVisitor stmts;

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

    ValueOrStmt VastCodeGenVisitor::VisitDeclStmt(clang::DeclStmt *stmt)
    {
        LLVM_DEBUG(llvm::dbgs() << "Visit DeclStmt\n");
        assert(stmt->isSingleDecl());
        return decls.Visit( *(stmt->decls().begin()) );
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
