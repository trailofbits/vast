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

        // Binary Operations

        ValueOrStmt VisitBinPtrMemD(clang::BinaryOperator *expr)
        {
            llvm_unreachable( "unsupported BinPtrMemD" );
        }

        ValueOrStmt VisitBinPtrMemI(clang::BinaryOperator *expr)
        {
            llvm_unreachable( "unsupported BinPtrMemI" );
        }

        ValueOrStmt VisitBinMul(clang::BinaryOperator *expr)
        {
            auto ty = expr->getType();
            if (ty->isIntegerType())
                return build_binary< MulIOp >(expr);
            llvm_unreachable( "unsupported BinMul" );
        }

        ValueOrStmt VisitBinDiv(clang::BinaryOperator *expr)
        {
            auto ty = expr->getType();
            if (ty->isUnsignedIntegerType())
                return build_binary< DivUOp >(expr);
            if (ty->isIntegerType())
                return build_binary< DivSOp >(expr);
            llvm_unreachable( "unsupported BinDiv" );
        }

        ValueOrStmt VisitBinRem(clang::BinaryOperator *expr)
        {
            auto ty = expr->getType();
            if (ty->isUnsignedIntegerType())
                return build_binary< RemUOp >(expr);
            if (ty->isIntegerType())
                return build_binary< RemSOp >(expr);
            llvm_unreachable( "unsupported BinRem" );
        }

        ValueOrStmt VisitBinAdd(clang::BinaryOperator *expr)
        {
            auto ty = expr->getType();
            if (ty->isIntegerType())
                return build_binary< AddIOp >(expr);
            llvm_unreachable( "unsupported addition type" );
        }

        ValueOrStmt VisitBinSub(clang::BinaryOperator *expr)
        {
            auto ty = expr->getType();
            if (ty->isIntegerType())
                return build_binary< SubIOp >(expr);
            llvm_unreachable( "unsupported BinSub" );
        }

        ValueOrStmt VisitBinShl(clang::BinaryOperator *expr)
        {
            llvm_unreachable( "unsupported BinShl" );
        }

        ValueOrStmt VisitBinShr(clang::BinaryOperator *expr)
        {
            llvm_unreachable( "unsupported BinShr" );
        }

        ValueOrStmt VisitBinLT(clang::BinaryOperator *expr)
        {
            auto ty = expr->getLHS()->getType();
            if (ty->isUnsignedIntegerType())
                return build_comparison< Predicate::ult >(expr);
            if (ty->isIntegerType())
                return build_comparison< Predicate::slt >(expr);
            llvm_unreachable( "unsupported BinLT" );
        }

        ValueOrStmt VisitBinGT(clang::BinaryOperator *expr)
        {
            auto ty = expr->getLHS()->getType();
            if (ty->isUnsignedIntegerType())
                return build_comparison< Predicate::ugt >(expr);
            if (ty->isIntegerType())
                return build_comparison< Predicate::sgt >(expr);
            llvm_unreachable( "unsupported BinGT" );
        }

        ValueOrStmt VisitBinLE(clang::BinaryOperator *expr)
        {
            auto ty = expr->getLHS()->getType();
            if (ty->isUnsignedIntegerType())
                return build_comparison< Predicate::ule >(expr);
            if (ty->isIntegerType())
                return build_comparison< Predicate::sle >(expr);
            llvm_unreachable( "unsupported BinLE" );
        }

        ValueOrStmt VisitBinGE(clang::BinaryOperator *expr)
        {
            auto ty = expr->getLHS()->getType();
            if (ty->isUnsignedIntegerType())
                return build_comparison< Predicate::uge >(expr);
            if (ty->isIntegerType())
                return build_comparison< Predicate::sge >(expr);
            llvm_unreachable( "unsupported BinGE" );
        }

        ValueOrStmt VisitBinEQ(clang::BinaryOperator *expr)
        {
            auto ty = expr->getLHS()->getType();
            if (ty->isIntegerType())
                return build_comparison< Predicate::eq >(expr);
            llvm_unreachable( "unsupported BinEQ" );
        }

        ValueOrStmt VisitBinNE(clang::BinaryOperator *expr)
        {
            auto ty = expr->getLHS()->getType();
            if (ty->isIntegerType())
                return build_comparison< Predicate::ne >(expr);
            llvm_unreachable( "unsupported BinNE" );
        }

        ValueOrStmt VisitBinAnd(clang::BinaryOperator *expr)
        {
            llvm_unreachable( "unsupported BinAnd" );
        }

        ValueOrStmt VisitBinXor(clang::BinaryOperator *expr)
        {
            llvm_unreachable( "unsupported BinXor" );
        }

        ValueOrStmt VisitBinOr(clang::BinaryOperator *expr)
        {
            llvm_unreachable( "unsupported BinOr" );
        }

        ValueOrStmt VisitBinLAnd(clang::BinaryOperator *expr)
        {
            llvm_unreachable( "unsupported BinLAnd" );
        }

        ValueOrStmt VisitBinLOr(clang::BinaryOperator *expr)
        {
            llvm_unreachable( "unsupported BinLOr" );
        }

        ValueOrStmt VisitBinAssign(clang::BinaryOperator *expr)
        {
            return build_binary< AssignOp >(expr);
        }

        // Compound Assignment Operations

        ValueOrStmt VisitBinMulAssign(clang::CompoundAssignOperator *expr)
        {
            auto ty = expr->getType();
            if (ty->isIntegerType())
                return build_binary< MulIAssignOp >(expr);
            llvm_unreachable( "unsupported BinMulAssign" );
        }

        ValueOrStmt VisitBinDivAssign(clang::CompoundAssignOperator *expr)
        {
            auto ty = expr->getType();
            if (ty->isUnsignedIntegerType())
                return build_binary< DivUAssignOp >(expr);
            if (ty->isIntegerType())
                return build_binary< DivSAssignOp >(expr);
            llvm_unreachable( "unsupported BinDivAssign" );
        }

        ValueOrStmt VisitBinRemAssign(clang::CompoundAssignOperator *expr)
        {
            auto ty = expr->getType();
            if (ty->isUnsignedIntegerType())
                return build_binary< RemUAssignOp >(expr);
            if (ty->isIntegerType())
                return build_binary< RemSAssignOp >(expr);
            llvm_unreachable( "unsupported BinRemAssign" );
        }

        ValueOrStmt VisitBinAddAssign(clang::CompoundAssignOperator *expr)
        {
            auto ty = expr->getType();
            if (ty->isIntegerType())
                return build_binary< AddIAssignOp >(expr);
            llvm_unreachable( "unsupported BinAddAssign" );
        }

        ValueOrStmt VisitBinSubAssign(clang::CompoundAssignOperator *expr)
        {
            auto ty = expr->getType();
            if (ty->isIntegerType())
                return build_binary< SubIAssignOp >(expr);
            llvm_unreachable( "unsupported BinSubAssign" );
        }

        ValueOrStmt VisitBinShlAssign(clang::CompoundAssignOperator *expr)
        {
            llvm_unreachable( "unsupported BinShlAssign" );
        }

        ValueOrStmt VisitBinShrAssign(clang::CompoundAssignOperator *expr)
        {
            llvm_unreachable( "unsupported BinShrAssign" );
        }

        ValueOrStmt VisitBinAndAssign(clang::CompoundAssignOperator *expr)
        {
            llvm_unreachable( "unsupported BinAndAssign" );
        }

        ValueOrStmt VisitBinOrAssign(clang::CompoundAssignOperator *expr)
        {
            llvm_unreachable( "unsupported BinOrAssign" );
        }

        ValueOrStmt VisitBinXorAssign(clang::CompoundAssignOperator *expr)
        {
            llvm_unreachable( "unsupported BinXorAssign" );
        }

        ValueOrStmt VisitBinComma(clang::BinaryOperator *expr)
        {
            llvm_unreachable( "unsupported BinComma" );
        }

        // Unary Operations

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
            llvm_unreachable( "unsupported UnaryAddrOf" );
        }

        ValueOrStmt VisitUnaryDeref(clang::UnaryOperator *expr)
        {
            llvm_unreachable( "unsupported UnaryDeref" );
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
            llvm_unreachable( "unsupported UnaryReal" );
        }

        ValueOrStmt VisitUnaryImag(clang::UnaryOperator *expr)
        {
            llvm_unreachable( "unsupported UnaryImag" );
        }

        ValueOrStmt VisitUnaryExtension(clang::UnaryOperator *expr)
        {
            llvm_unreachable( "unsupported UnaryExtension" );
        }

        ValueOrStmt VisitUnaryCoawait(clang::UnaryOperator *expr)
        {
            llvm_unreachable( "unsupported UnaryCoawait" );
        }

        // Assembky Statements

        ValueOrStmt VisitAsmStmt(clang::AsmStmt *stmt)
        {
            llvm_unreachable( "unsupported AsmStmt" );
        }

        ValueOrStmt VisitGCCAsmStmt(clang::GCCAsmStmt *stmt)
        {
            llvm_unreachable( "unsupported GCCAsmStmt" );
        }

        ValueOrStmt VisitMSAsmStmt(clang::MSAsmStmt *stmt)
        {
            llvm_unreachable( "unsupported MSAsmStmt" );
        }

        ValueOrStmt VisitCoroutineBodyStmt(clang::CoroutineBodyStmt *stmt)
        {
            llvm_unreachable( "unsupported CoroutineBodyStmt" );
        }

        ValueOrStmt VisitCoreturnStmt(clang::CoreturnStmt *stmt)
        {
            llvm_unreachable( "unsupported CoreturnStmt" );
        }

        ValueOrStmt VisitCoroutineSuspendExpr(clang::CoroutineSuspendExpr *expr)
        {
            llvm_unreachable( "unsupported CoroutineSuspendExpr" );
        }

        ValueOrStmt VisitCoawaitExpr(clang::CoawaitExpr *expr)
        {
            llvm_unreachable( "unsupported CoawaitExpr" );
        }

        ValueOrStmt VisitCoyieldExpr(clang::CoyieldExpr *expr)
        {
            llvm_unreachable( "unsupported CoyieldExpr" );
        }

        ValueOrStmt VisitDependentCoawaitExpr(clang::DependentCoawaitExpr *expr)
        {
            llvm_unreachable( "unsupported DependentCoawaitExpr" );
        }

        ValueOrStmt VisitAttributedStmt(clang::AttributedStmt *stmt)
        {
            llvm_unreachable( "unsupported AttributedStmt" );
        }

        // Statements

        ValueOrStmt VisitBreakStmt(clang::BreakStmt *stmt)
        {
            llvm_unreachable( "unsupported BreakStmt" );
        }

        ValueOrStmt VisitCXXCatchStmt(clang::CXXCatchStmt *stmt)
        {
            llvm_unreachable( "unsupported CXXCatchStmt" );
        }

        ValueOrStmt VisitCXXForRangeStmt(clang::CXXForRangeStmt *stmt)
        {
            llvm_unreachable( "unsupported CXXForRangeStmt" );
        }

        ValueOrStmt VisitCXXTryStmt(clang::CXXTryStmt *stmt)
        {
            llvm_unreachable( "unsupported CXXTryStmt" );
        }

        ValueOrStmt VisitCapturedStmt(clang::CapturedStmt *stmt)
        {
            llvm_unreachable( "unsupported CapturedStmt" );
        }

        ValueOrStmt VisitCompoundStmt(clang::CompoundStmt *stmt)
        {
            ScopedInsertPoint builder_scope(builder);

            auto loc = builder.getLocation(stmt->getSourceRange());

            ScopeOp scope = builder.create< ScopeOp >(loc);
            auto &body = scope.body();
            body.push_back( new mlir::Block() );
            builder.setInsertionPointToStart( &body.front() );

            for (auto s : stmt->body()) {
                Visit(s);
            }

            auto &lastblock = body.back();
            if (lastblock.empty() || lastblock.back().isKnownNonTerminator()) {
                builder.setInsertionPointToEnd(&lastblock);
                builder.create< ScopeEndOp >(loc);
            }

            return scope;
        }

        ValueOrStmt VisitContinueStmt(clang::ContinueStmt *stmt)
        {
            llvm_unreachable( "unsupported ContinueStmt" );
        }

        ValueOrStmt VisitDeclStmt(clang::DeclStmt *stmt);

        ValueOrStmt VisitDoStmt(clang::DoStmt *stmt)
        {
            llvm_unreachable( "unsupported DoStmt" );
        }

        // Expressions

        ValueOrStmt VisitAbstractConditionalOperator(clang::AbstractConditionalOperator *stmt)
        {
            llvm_unreachable( "unsupported AbstractConditionalOperator" );
        }

        ValueOrStmt VisitBinaryConditionalOperator(clang::BinaryConditionalOperator *stmt)
        {
            llvm_unreachable( "unsupported BinaryConditionalOperator" );
        }

        ValueOrStmt VisitConditionalOperator(clang::ConditionalOperator *stmt)
        {
            llvm_unreachable( "unsupported ConditionalOperator" );
        }

        ValueOrStmt VisitAddrLabelExpr(clang::AddrLabelExpr *expr)
        {
            llvm_unreachable( "unsupported AddrLabelExpr" );
        }

        ValueOrStmt VisitArraySubscriptExpr(clang::ArraySubscriptExpr *expr)
        {
            llvm_unreachable( "unsupported ArraySubscriptExpr" );
        }

        ValueOrStmt VisitArrayTypeTraitExpr(clang::ArrayTypeTraitExpr *expr)
        {
            llvm_unreachable( "unsupported ArrayTypeTraitExpr" );
        }

        ValueOrStmt VisitAsTypeExpr(clang::AsTypeExpr *expr)
        {
            llvm_unreachable( "unsupported AsTypeExpr" );
        }

        ValueOrStmt VisitAtomicExpr(clang::AtomicExpr *expr)
        {
            llvm_unreachable( "unsupported AtomicExpr" );
        }

        ValueOrStmt VisitBlockExpr(clang::BlockExpr *expr)
        {
            llvm_unreachable( "unsupported BlockExpr" );
        }

        ValueOrStmt VisitCXXBindTemporaryExpr(clang::CXXBindTemporaryExpr *expr)
        {
            llvm_unreachable( "unsupported CXXBindTemporaryExpr" );
        }

        ValueOrStmt VisitCXXBoolLiteralExpr(const clang::CXXBoolLiteralExpr *lit)
        {
            return VisitLiteral(lit->getValue(), lit);
        }

        ValueOrStmt VisitCXXConstructExpr(clang::CXXConstructExpr *expr)
        {
            llvm_unreachable( "unsupported CXXConstructExpr" );
        }

        ValueOrStmt VisitCXXTemporaryObjectExpr(clang::CXXTemporaryObjectExpr *expr)
        {
            llvm_unreachable( "unsupported CXXTemporaryObjectExpr" );
        }

        ValueOrStmt VisitCXXDefaultArgExpr(clang::CXXDefaultArgExpr *expr)
        {
            llvm_unreachable( "unsupported CXXDefaultArgExpr" );
        }

        ValueOrStmt VisitCXXDefaultInitExpr(clang::CXXDefaultInitExpr *expr)
        {
            llvm_unreachable( "unsupported CXXDefaultInitExpr" );
        }

        ValueOrStmt VisitCXXDeleteExpr(clang::CXXDeleteExpr *expr)
        {
            llvm_unreachable( "unsupported CXXDeleteExpr" );
        }

        ValueOrStmt VisitCXXDependentScopeMemberExpr(clang::CXXDependentScopeMemberExpr *expr)
        {
            llvm_unreachable( "unsupported CXXDependentScopeMemberExpr" );
        }

        ValueOrStmt VisitCXXNewExpr(clang::CXXNewExpr *expr)
        {
            llvm_unreachable( "unsupported CXXNewExpr" );
        }

        ValueOrStmt VisitCXXNoexceptExpr(clang::CXXNoexceptExpr *expr)
        {
            llvm_unreachable( "unsupported CXXNoexceptExpr" );
        }

        ValueOrStmt VisitCXXNullPtrLiteralExpr(clang::CXXNullPtrLiteralExpr *expr)
        {
            llvm_unreachable( "unsupported CXXNullPtrLiteralExpr" );
        }

        ValueOrStmt VisitCXXPseudoDestructorExpr(clang::CXXPseudoDestructorExpr *expr)
        {
            llvm_unreachable( "unsupported CXXPseudoDestructorExpr" );
        }

        ValueOrStmt VisitCXXScalarValueInitExpr(clang::CXXScalarValueInitExpr *expr)
        {
            llvm_unreachable( "unsupported CXXScalarValueInitExpr" );
        }

        ValueOrStmt VisitCXXStdInitializerListExpr(clang::CXXStdInitializerListExpr *expr)
        {
            llvm_unreachable( "unsupported CXXStdInitializerListExpr" );
        }

        ValueOrStmt VisitCXXThisExpr(clang::CXXThisExpr *expr)
        {
            llvm_unreachable( "unsupported CXXThisExpr" );
        }

        ValueOrStmt VisitCXXThrowExpr(clang::CXXThrowExpr *expr)
        {
            llvm_unreachable( "unsupported CXXThrowExpr" );
        }

        ValueOrStmt VisitCXXTypeidExpr(clang::CXXTypeidExpr *expr)
        {
            llvm_unreachable( "unsupported CXXTypeidExpr" );
        }

        ValueOrStmt VisitCXXUnresolvedConstructExpr(clang::CXXUnresolvedConstructExpr *expr)
        {
            llvm_unreachable( "unsupported CXXUnresolvedConstructExpr" );
        }

        ValueOrStmt VisitCXXUuidofExpr(clang::CXXUuidofExpr *expr)
        {
            llvm_unreachable( "unsupported CXXUuidofExpr" );
        }

        ValueOrStmt VisitCallExpr(clang::CallExpr *expr)
        {
            llvm_unreachable( "unsupported CallExpr" );
        }

        ValueOrStmt VisitCUDAKernelCallExpr(clang::CUDAKernelCallExpr *expr)
        {
            llvm_unreachable( "unsupported CUDAKernelCallExpr" );
        }

        ValueOrStmt VisitCXXMemberCallExpr(clang::CXXMemberCallExpr *expr)
        {
            llvm_unreachable( "unsupported CXXMemberCallExpr" );
        }

        ValueOrStmt VisitCXXOperatorCallExpr(clang::CXXOperatorCallExpr *expr)
        {
            llvm_unreachable( "unsupported CXXOperatorCallExpr" );
        }

        ValueOrStmt VisitUserDefinedLiteral(clang::UserDefinedLiteral *lit)
        {
            llvm_unreachable( "unsupported UserDefinedLiteral" );
        }

        ValueOrStmt VisitCStyleCastExpr(clang::CStyleCastExpr *expr)
        {
            return build_cast< CStyleCastOp >( expr->getSubExpr(), expr->getType(), cast_kind(expr) );
        }

        ValueOrStmt VisitCXXFunctionalCastExpr(clang::CXXFunctionalCastExpr *expr)
        {
            llvm_unreachable( "unsupported CXXFunctionalCastExpr" );
        }

        ValueOrStmt VisitCXXConstCastExpr(clang::CXXConstCastExpr *expr)
        {
            llvm_unreachable( "unsupported CXXConstCastExpr" );
        }

        ValueOrStmt VisitCXXDynamicCastExpr(clang::CXXDynamicCastExpr *expr)
        {
            llvm_unreachable( "unsupported CXXDynamicCastExpr" );
        }

        ValueOrStmt VisitCXXReinterpretCastExpr(clang::CXXReinterpretCastExpr *expr)
        {
            llvm_unreachable( "unsupported CXXReinterpretCastExpr" );
        }

        ValueOrStmt VisitCXXStaticCastExpr(clang::CXXStaticCastExpr *expr)
        {
            llvm_unreachable( "unsupported CXXStaticCastExpr" );
        }

        ValueOrStmt VisitObjCBridgedCastExpr(clang::ObjCBridgedCastExpr *expr)
        {
            llvm_unreachable( "unsupported ObjCBridgedCastExpr" );
        }

        ValueOrStmt VisitImplicitCastExpr(clang::ImplicitCastExpr *expr)
        {
            return build_cast< ImplicitCastOp >( expr->getSubExpr(), expr->getType(), cast_kind(expr) );
        }

        ValueOrStmt VisitCharacterLiteral(clang::CharacterLiteral *lit)
        {
            llvm_unreachable( "unsupported CharacterLiteral" );
        }

        ValueOrStmt VisitChooseExpr(clang::ChooseExpr *expr)
        {
            llvm_unreachable( "unsupported ChooseExpr" );
        }

        ValueOrStmt VisitCompoundLiteralExpr(clang::CompoundLiteralExpr *expr)
        {
            llvm_unreachable( "unsupported CompoundLiteralExpr" );
        }

        ValueOrStmt VisitConvertVectorExpr(clang::ConvertVectorExpr *expr)
        {
            llvm_unreachable( "unsupported ConvertVectorExpr" );
        }

        ValueOrStmt VisitDeclRefExpr(clang::DeclRefExpr *expr)
        {
            auto loc = builder.getLocation(expr->getSourceRange());

            // TODO(Heno): deal with function declaration

            // TODO(Heno): deal with enum constant declaration

            auto named = expr->getDecl()->getUnderlyingDecl();
            auto rty = types.convert(expr->getType());
            return builder.create_value< DeclRefOp >( loc, rty, named->getNameAsString() );
        }

        ValueOrStmt VisitDependentScopeDeclRefExpr(clang::DependentScopeDeclRefExpr *expr)
        {
            llvm_unreachable( "unsupported DependentScopeDeclRefExpr" );
        }

        ValueOrStmt VisitDesignatedInitExpr(clang::DesignatedInitExpr *expr)
        {
            llvm_unreachable( "unsupported DesignatedInitExpr" );
        }

        ValueOrStmt VisitExprWithCleanups(clang::ExprWithCleanups *expr)
        {
            llvm_unreachable( "unsupported ExprWithCleanups" );
        }

        ValueOrStmt VisitExpressionTraitExpr(clang::ExpressionTraitExpr *expr)
        {
            llvm_unreachable( "unsupported ExpressionTraitExpr" );
        }

        ValueOrStmt VisitExtVectorElementExpr(clang::ExtVectorElementExpr *expr)
        {
            llvm_unreachable( "unsupported ExtVectorElementExpr" );
        }

        ValueOrStmt VisitFloatingLiteral(clang::FloatingLiteral *lit)
        {
            llvm_unreachable( "unsupported FloatingLiteral" );
        }

        ValueOrStmt VisitFunctionParmPackExpr(clang::FunctionParmPackExpr *expr)
        {
            llvm_unreachable( "unsupported FunctionParmPackExpr" );
        }

        ValueOrStmt VisitGNUNullExpr(clang::GNUNullExpr *expr)
        {
            llvm_unreachable( "unsupported GNUNullExpr" );
        }

        ValueOrStmt VisitGenericSelectionExpr(clang::GenericSelectionExpr *expr)
        {
            llvm_unreachable( "unsupported GenericSelectionExpr" );
        }

        ValueOrStmt VisitImaginaryLiteral(clang::ImaginaryLiteral *lit)
        {
            llvm_unreachable( "unsupported ImaginaryLiteral" );
        }

        ValueOrStmt VisitImplicitValueInitExpr(clang::ImplicitValueInitExpr *expr)
        {
            llvm_unreachable( "unsupported ImplicitValueInitExpr" );
        }

        ValueOrStmt VisitInitListExpr(clang::InitListExpr *expr)
        {
            llvm_unreachable( "unsupported InitListExpr" );
        }

        ValueOrStmt VisitIntegerLiteral(const clang::IntegerLiteral *lit)
        {
            return VisitLiteral(lit->getValue().getSExtValue(), lit);
        }

        ValueOrStmt VisitLambdaExpr(clang::LambdaExpr *expr)
        {
            llvm_unreachable( "unsupported LambdaExpr" );
        }

        ValueOrStmt VisitMSPropertyRefExpr(clang::MSPropertyRefExpr *expr)
        {
            llvm_unreachable( "unsupported MSPropertyRefExpr" );
        }

        ValueOrStmt VisitMaterializeTemporaryExpr(clang::MaterializeTemporaryExpr *expr)
        {
            llvm_unreachable( "unsupported MaterializeTemporaryExpr" );
        }

        ValueOrStmt VisitMemberExpr(clang::MemberExpr *expr)
        {
            llvm_unreachable( "unsupported MemberExpr" );
        }

        ValueOrStmt VisitObjCArrayLiteral(clang::ObjCArrayLiteral *expr)
        {
            llvm_unreachable( "unsupported ObjCArrayLiteral" );
        }

        ValueOrStmt VisitObjCBoolLiteralExpr(clang::ObjCBoolLiteralExpr *expr)
        {
            llvm_unreachable( "unsupported ObjCBoolLiteralExpr" );
        }

        ValueOrStmt VisitObjCBoxedExpr(clang::ObjCBoxedExpr *expr)
        {
            llvm_unreachable( "unsupported ObjCBoxedExpr" );
        }

        ValueOrStmt VisitObjCDictionaryLiteral(clang::ObjCDictionaryLiteral *lit)
        {
            llvm_unreachable( "unsupported ObjCDictionaryLiteral" );
        }

        ValueOrStmt VisitObjCEncodeExpr(clang::ObjCEncodeExpr *expr)
        {
            llvm_unreachable( "unsupported ObjCEncodeExpr" );
        }

        ValueOrStmt VisitObjCIndirectCopyRestoreExpr(clang::ObjCIndirectCopyRestoreExpr *expr)
        {
            llvm_unreachable( "unsupported ObjCIndirectCopyRestoreExpr" );
        }

        ValueOrStmt VisitObjCIsaExpr(clang::ObjCIsaExpr *expr)
        {
            llvm_unreachable( "unsupported ObjCIsaExpr" );
        }

        ValueOrStmt VisitObjCIvarRefExpr(clang::ObjCIvarRefExpr *expr)
        {
            llvm_unreachable( "unsupported ObjCIvarRefExpr" );
        }

        ValueOrStmt VisitObjCMessageExpr(clang::ObjCMessageExpr *expr)
        {
            llvm_unreachable( "unsupported ObjCMessageExpr" );
        }

        ValueOrStmt VisitObjCPropertyRefExpr(clang::ObjCPropertyRefExpr *expr)
        {
            llvm_unreachable( "unsupported ObjCPropertyRefExpr" );
        }

        ValueOrStmt VisitObjCProtocolExpr(clang::ObjCProtocolExpr *expr)
        {
            llvm_unreachable( "unsupported ObjCProtocolExpr" );
        }

        ValueOrStmt VisitObjCSelectorExpr(clang::ObjCSelectorExpr *expr)
        {
            llvm_unreachable( "unsupported ObjCSelectorExpr" );
        }

        ValueOrStmt VisitObjCStringLiteral(clang::ObjCStringLiteral *lit)
        {
            llvm_unreachable( "unsupported ObjCStringLiteral" );
        }

        ValueOrStmt VisitObjCSubscriptRefExpr(clang::ObjCSubscriptRefExpr *expr)
        {
            llvm_unreachable( "unsupported ObjCSubscriptRefExpr" );
        }

        ValueOrStmt VisitOffsetOfExpr(clang::OffsetOfExpr *expr)
        {
            llvm_unreachable( "unsupported OffsetOfExpr" );
        }

        ValueOrStmt VisitOpaqueValueExpr(clang::OpaqueValueExpr *expr)
        {
            llvm_unreachable( "unsupported OpaqueValueExpr" );
        }

        ValueOrStmt VisitOverloadExpr(clang::OverloadExpr *expr)
        {
            llvm_unreachable( "unsupported OverloadExpr" );
        }

        ValueOrStmt VisitUnresolvedLookupExpr(clang::UnresolvedLookupExpr *expr)
        {
            llvm_unreachable( "unsupported UnresolvedLookupExpr" );
        }

        ValueOrStmt VisitUnresolvedMemberExpr(clang::UnresolvedMemberExpr *expr)
        {
            llvm_unreachable( "unsupported UnresolvedMemberExpr" );
        }

        ValueOrStmt VisitPackExpansionExpr(clang::PackExpansionExpr *expr)
        {
            llvm_unreachable( "unsupported PackExpansionExpr" );
        }

        ValueOrStmt VisitParenExpr(clang::ParenExpr *expr)
        {
            llvm_unreachable( "unsupported ParenExpr" );
        }

        ValueOrStmt VisitParenListExpr(clang::ParenListExpr *expr)
        {
            llvm_unreachable( "unsupported ParenListExpr" );
        }

        ValueOrStmt VisitPredefinedExpr(clang::PredefinedExpr *expr)
        {
            llvm_unreachable( "unsupported PredefinedExpr" );
        }

        ValueOrStmt VisitPseudoObjectExpr(clang::PseudoObjectExpr *expr)
        {
            llvm_unreachable( "unsupported PseudoObjectExpr" );
        }

        ValueOrStmt VisitShuffleVectorExpr(clang::ShuffleVectorExpr *expr)
        {
            llvm_unreachable( "unsupported ShuffleVectorExpr" );
        }

        ValueOrStmt VisitSizeOfPackExpr(clang::SizeOfPackExpr *expr)
        {
            llvm_unreachable( "unsupported SizeOfPackExpr" );
        }

        ValueOrStmt VisitStmtExpr(clang::StmtExpr *expr)
        {
            llvm_unreachable( "unsupported StmtExpr" );
        }

        ValueOrStmt VisitStringLiteral(clang::StringLiteral *lit)
        {
            llvm_unreachable( "unsupported StringLiteral" );
        }

        ValueOrStmt VisitSubstNonTypeTemplateParmExpr(clang::SubstNonTypeTemplateParmExpr *expr)
        {
            llvm_unreachable( "unsupported SubstNonTypeTemplateParmExpr" );
        }

        ValueOrStmt VisitSubstNonTypeTemplateParmPackExpr(clang::SubstNonTypeTemplateParmPackExpr *expr)
        {
            llvm_unreachable( "unsupported SubstNonTypeTemplateParmPackExpr" );
        }

        ValueOrStmt VisitTypeTraitExpr(clang::TypeTraitExpr *expr)
        {
            llvm_unreachable( "unsupported TypeTraitExpr" );
        }

        ValueOrStmt VisitUnaryExprOrTypeTraitExpr(clang::UnaryExprOrTypeTraitExpr *expr)
        {
            llvm_unreachable( "unsupported UnaryExprOrTypeTraitExpr" );
        }

        ValueOrStmt VisitVAArgExpr(clang::VAArgExpr *expr)
        {
            llvm_unreachable( "unsupported VAArgExpr" );
        }

        // Statements

        ValueOrStmt VisitForStmt(clang::ForStmt *stmt)
        {
            auto loc = builder.getLocation(stmt->getSourceRange());

            auto init_builder = make_nonterminated_scope_builder(stmt->getInit());
            auto cond_builder = make_nonterminated_scope_builder(stmt->getCond());
            auto incr_builder = make_nonterminated_scope_builder(stmt->getInc());
            auto body_builder = make_scope_builder(stmt->getBody());

            return builder.create< ForOp >(loc, init_builder, cond_builder, incr_builder, body_builder);
        }

        ValueOrStmt VisitGotoStmt(clang::GotoStmt *stmt)
        {
            llvm_unreachable( "unsupported GotoStmt" );
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

        ValueOrStmt VisitIndirectGotoStmt(clang::IndirectGotoStmt *stmt)
        {
            llvm_unreachable( "unsupported IndirectGotoStmt" );
        }

        ValueOrStmt VisitLabelStmt(clang::LabelStmt *stmt)
        {
            llvm_unreachable( "unsupported LabelStmt" );
        }

        ValueOrStmt VisitMSDependentExistsStmt(clang::MSDependentExistsStmt *stmt)
        {
            llvm_unreachable( "unsupported MSDependentExistsStmt" );
        }

        ValueOrStmt VisitNullStmt(clang::NullStmt *stmt)
        {
            llvm_unreachable( "unsupported NullStmt" );
        }

        ValueOrStmt VisitOMPBarrierDirective(clang::OMPBarrierDirective *dir)
        {
            llvm_unreachable( "unsupported OMPBarrierDirective" );
        }

        ValueOrStmt VisitOMPCriticalDirective(clang::OMPCriticalDirective *dir)
        {
            llvm_unreachable( "unsupported OMPCriticalDirective" );
        }

        ValueOrStmt VisitOMPFlushDirective(clang::OMPFlushDirective *dir)
        {
            llvm_unreachable( "unsupported OMPFlushDirective" );
        }

        ValueOrStmt VisitOMPForDirective(clang::OMPForDirective *dir)
        {
            llvm_unreachable( "unsupported OMPForDirective" );
        }

        ValueOrStmt VisitOMPMasterDirective(clang::OMPMasterDirective *dir)
        {
            llvm_unreachable( "unsupported OMPMasterDirective" );
        }

        ValueOrStmt VisitOMPParallelDirective(clang::OMPParallelDirective *dir)
        {
            llvm_unreachable( "unsupported OMPParallelDirective" );
        }

        ValueOrStmt VisitOMPParallelForDirective(clang::OMPParallelForDirective *dir)
        {
            llvm_unreachable( "unsupported OMPParallelForDirective" );
        }

        ValueOrStmt VisitOMPParallelSectionsDirective(clang::OMPParallelSectionsDirective *dir)
        {
            llvm_unreachable( "unsupported OMPParallelSectionsDirective" );
        }

        ValueOrStmt VisitOMPSectionDirective(clang::OMPSectionDirective *dir)
        {
            llvm_unreachable( "unsupported OMPSectionDirective" );
        }

        ValueOrStmt VisitOMPSectionsDirective(clang::OMPSectionsDirective *dir)
        {
            llvm_unreachable( "unsupported OMPSectionsDirective" );
        }

        ValueOrStmt VisitOMPSimdDirective(clang::OMPSimdDirective *dir)
        {
            llvm_unreachable( "unsupported OMPSimdDirective" );
        }

        ValueOrStmt VisitOMPSingleDirective(clang::OMPSingleDirective *dir)
        {
            llvm_unreachable( "unsupported OMPSingleDirective" );
        }

        ValueOrStmt VisitOMPTaskDirective(clang::OMPTaskDirective *dir)
        {
            llvm_unreachable( "unsupported OMPTaskDirective" );
        }

        ValueOrStmt VisitOMPTaskwaitDirective(clang::OMPTaskwaitDirective *dir)
        {
            llvm_unreachable( "unsupported OMPTaskwaitDirective" );
        }

        ValueOrStmt VisitOMPTaskyieldDirective(clang::OMPTaskyieldDirective *dir)
        {
            llvm_unreachable( "unsupported OMPTaskyieldDirective" );
        }

        ValueOrStmt VisitObjCAtCatchStmt(clang::ObjCAtCatchStmt *stmt)
        {
            llvm_unreachable( "unsupported ObjCAtCatchStmt" );
        }

        ValueOrStmt VisitObjCAtFinallyStmt(clang::ObjCAtFinallyStmt *stmt)
        {
            llvm_unreachable( "unsupported ObjCAtFinallyStmt" );
        }

        ValueOrStmt VisitObjCAtSynchronizedStmt(clang::ObjCAtSynchronizedStmt *stmt)
        {
            llvm_unreachable( "unsupported ObjCAtSynchronizedStmt" );
        }

        ValueOrStmt VisitObjCAtThrowStmt(clang::ObjCAtThrowStmt *stmt)
        {
            llvm_unreachable( "unsupported ObjCAtThrowStmt" );
        }

        ValueOrStmt VisitObjCAtTryStmt(clang::ObjCAtTryStmt *stmt)
        {
            llvm_unreachable( "unsupported ObjCAtTryStmt" );
        }

        ValueOrStmt VisitObjCAutoreleasePoolStmt(clang::ObjCAutoreleasePoolStmt *stmt)
        {
            llvm_unreachable( "unsupported ObjCAutoreleasePoolStmt" );
        }

        ValueOrStmt VisitObjCForCollectionStmt(clang::ObjCForCollectionStmt *stmt)
        {
            llvm_unreachable( "unsupported ObjCForCollectionStmt" );
        }

        ValueOrStmt VisitReturnStmt(clang::ReturnStmt *stmt)
        {
            auto loc = builder.getLocation(stmt->getSourceRange());
            if (stmt->getRetValue()) {
                auto val = Visit(stmt->getRetValue());
                return builder.create< ReturnOp >(loc, val);
            } else {
                auto val = builder.create_void(loc);
                return builder.create< ReturnOp >(loc, val);
            }
        }

        ValueOrStmt VisitSEHExceptStmt(clang::SEHExceptStmt *stmt)
        {
            llvm_unreachable( "unsupported SEHExceptStmt" );
        }

        ValueOrStmt VisitSEHFinallyStmt(clang::SEHFinallyStmt *stmt)
        {
            llvm_unreachable( "unsupported SEHFinallyStmt" );
        }

        ValueOrStmt VisitSEHLeaveStmt(clang::SEHLeaveStmt *stmt)
        {
            llvm_unreachable( "unsupported SEHLeaveStmt" );
        }

        ValueOrStmt VisitSEHTryStmt(clang::SEHTryStmt *stmt)
        {
            llvm_unreachable( "unsupported SEHTryStmt" );
        }

        ValueOrStmt VisitSwitchCase(clang::SwitchCase *stmt)
        {
            llvm_unreachable( "unsupported SwitchCase" );
        }

        ValueOrStmt VisitCaseStmt(clang::CaseStmt *stmt)
        {
            llvm_unreachable( "unsupported CaseStmt" );
        }

        ValueOrStmt VisitDefaultStmt(clang::DefaultStmt *stmt)
        {
            llvm_unreachable( "unsupported DefaultStmt" );
        }

        ValueOrStmt VisitSwitchStmt(clang::SwitchStmt *stmt)
        {
            llvm_unreachable( "unsupported SwitchStmt" );
        }

        ValueOrStmt VisitWhileStmt(clang::WhileStmt *stmt)
        {
            auto loc = builder.getLocation(stmt->getSourceRange());
            auto body_builder = make_scope_builder(stmt->getBody());

            auto cond = Visit(stmt->getCond());
            return builder.create< WhileOp >(loc, cond, body_builder);
        }

        template< typename Value, typename Literal >
        ValueOrStmt VisitLiteral(Value val, Literal lit)
        {
            auto type = types.convert(lit->getType());
            auto loc = builder.getLocation(lit->getSourceRange());
            return builder.constant(loc, type, val);
        }

        ValueOrStmt VisitBuiltinBitCastExpr(clang::BuiltinBitCastExpr *expr)
        {
            return build_cast< BuiltinBitCastOp >( expr->getSubExpr(), expr->getType(), cast_kind(expr) );
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

        mlir::OwningModuleRef mod(
            mlir::ModuleOp::create(
                mlir::FileLineColLoc::get(input->getBufferIdentifier(), /* line */ 0, /* column */ 0, ctx)
            )
        );

        auto ast = clang::tooling::buildASTFromCode(input->getBuffer());

        VastCodeGen codegen(*ctx, mod, ast->getASTContext());
        codegen.HandleTranslationUnit(ast->getASTContext());

        // TODO(Heno): verify module
        return mod;
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
