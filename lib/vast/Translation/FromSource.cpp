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

#include "vast/Util/Types.hpp"
#include "vast/Translation/Types.hpp"
#include "vast/Translation/Expr.hpp"
#include "vast/Dialect/HighLevel/HighLevelDialect.hpp"
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

namespace vast::hl
{
    using logical_result = mlir::LogicalResult;

    using Stmt = mlir::Operation*;
    using ValueOrStmt = std::variant< mlir::Value, Stmt >;

    bool check(const ValueOrStmt &val)
    {
        return std::visit([] (const auto &v) { return bool(v); }, val);
    }

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
            return mlir::FileLineColLoc::get(file, loc.getLine(), loc.getColumn());

        }

        static inline auto to_value = [] (ValueOrStmt v) -> Value
        {
            return std::get< Value >(v);
        };

        static inline auto convert = overloaded{ to_value, identity };

        template< typename Op, typename ...Args >
        auto make(Args &&... args)
        {
            return builder.create< Op >( convert( std::forward< Args >(args) )... );
        }

        template< typename Op, typename ...Args >
        Value make_value(Args &&... args)
        {
            return make< Op >( std::forward< Args >(args)... );
        }

        template< typename Op, typename ...Args >
        Stmt make_stmt(Args &&... args)
        {
            return make< Op >( std::forward< Args >(args)... );
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

        mlir::Value bool_value(mlir::Location loc, bool value)
        {
            return make< ConstantOp >(loc, bool_type(), value);
        }

        mlir::Value true_value(mlir::Location loc)  { return bool_value(loc, true);  }
        mlir::Value false_value(mlir::Location loc) { return bool_value(loc, false); }

        mlir::Value constant(mlir::Location loc, mlir::Type ty, bool value)
        {
            CHECK(ty.isa< BoolType >(), "mismatched boolean constant type");
            return bool_value(loc, value);
        }

        mlir::Value constant(mlir::Location loc, mlir::Type ty, llvm::APInt value)
        {
            auto make_constant = [&] (auto ity) { return make< ConstantOp >(loc, ity, value); };
            return util::dispatch< integer_types, mlir::Value >(ty, make_constant);
        }

        mlir::Value constant(mlir::Location loc, mlir::Type ty, unsigned int value)
        {
            return constant(loc, ty, llvm::APInt(32, value));
        }

        mlir::Value constant(mlir::Location loc, mlir::Type ty, llvm::APFloat value)
        {
            auto make_constant = [&] (auto ity) { return make< ConstantOp >(loc, ity, value); };
            return util::dispatch< floating_types, mlir::Value >(ty, make_constant);
        }

        mlir::Value constant(mlir::Location loc, mlir::Type ty, llvm::StringRef value)
        {
            CHECK( ty.isa< ArrayType >(), "string constant must have array type" );
            return make< ConstantOp >(loc, ty.cast< ArrayType >(), value);
        }

        BoolType bool_type() { return BoolType::get(&mctx); }

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

    struct VastCodeGenVisitor
        : clang::StmtVisitor< VastCodeGenVisitor, ValueOrStmt >
        , clang::DeclVisitor< VastCodeGenVisitor, ValueOrStmt >
    {
        VastCodeGenVisitor(mlir::MLIRContext &mctx, mlir::OwningModuleRef &mod, clang::ASTContext &actx)
            : mod(mod)
            , builder(mctx, mod, actx)
            , types(&mctx, actx)
        {}

        using clang::StmtVisitor< VastCodeGenVisitor, ValueOrStmt >::Visit;
        using clang::DeclVisitor< VastCodeGenVisitor, ValueOrStmt >::Visit;

        template< typename Op, typename ...Args >
        auto make(Args &&... args)
        {
            return builder.make< Op >( std::forward< Args >(args)... );
        }

        template< typename Op, typename ...Args >
        Value make_value(Args &&... args)
        {
            return builder.make_value< Op >( std::forward< Args >(args)... );
        }

        template< typename Op, typename ...Args >
        Stmt make_stmt(Args &&... args)
        {
            return builder.make_stmt< Op >( std::forward< Args >(args)... );
        }

        template< typename Op >
        ValueOrStmt make_bin(clang::BinaryOperator *expr)
        {
            auto lhs = Visit(expr->getLHS());
            auto rhs = Visit(expr->getRHS());
            auto loc = builder.getEndLocation(expr->getSourceRange());
            auto res = make< Op >( loc, lhs, rhs );

            if constexpr ( std::is_convertible_v< decltype(res), Value > ) {
                return Value(res);
            } else {
                return res;
            }
        }

        template< typename Op >
        ValueOrStmt make_ibin(clang::BinaryOperator *expr)
        {
            auto ty = expr->getType();
            if (ty->isIntegerType())
                return make_bin< Op >(expr);
            return Value();
        }

        template< typename UOp, typename SOp >
        ValueOrStmt make_ibin(clang::BinaryOperator *expr)
        {
            auto ty = expr->getType();
            if (ty->isUnsignedIntegerType())
                return make_bin< UOp >(expr);
            if (ty->isIntegerType())
                return make_bin< SOp >(expr);
            return Value();
        }

        template< Predicate pred >
        Value make_cmp(clang::BinaryOperator *expr)
        {
            auto lhs = Visit(expr->getLHS());
            auto rhs = Visit(expr->getRHS());
            auto loc = builder.getEndLocation(expr->getSourceRange());
            auto res = builder.bool_type();
            return make_value< CmpOp >( loc, res, pred, lhs, rhs );
        }

        template< Predicate pred >
        Value make_icmp(clang::BinaryOperator *expr)
        {
            auto ty = expr->getLHS()->getType();
            if (ty->isIntegerType())
                return make_cmp< pred >(expr);
            return Value();
        }

        template< Predicate upred, Predicate spred >
        Value make_icmp(clang::BinaryOperator *expr)
        {
            auto ty = expr->getLHS()->getType();
            if (ty->isUnsignedIntegerType())
                return make_cmp< upred >(expr);
            if (ty->isIntegerType())
                return make_cmp< spred >(expr);
            return Value();
        }


        template< typename Op >
        ValueOrStmt make_type_preserving_unary(clang::UnaryOperator *expr)
        {
            auto loc = builder.getEndLocation(expr->getSourceRange());
            auto arg = Visit(expr->getSubExpr());
            return make_value< Op >( loc, arg );
        }

        template< typename Op >
        ValueOrStmt make_inplace_unary(clang::UnaryOperator *expr)
        {
            auto loc = builder.getLocation(expr->getSourceRange());
            auto arg = Visit(expr->getSubExpr());
            return make< Op >( loc, arg );
        }

        template< typename Op >
        ValueOrStmt make_unary(clang::UnaryOperator *expr)
        {
            auto loc = builder.getLocation(expr->getSourceRange());
            auto rty = types.convert(expr->getType());
            auto arg = Visit(expr->getSubExpr());
            return make_value< Op >( loc, rty, arg );
        }

        template< typename Cast >
        ValueOrStmt make_cast(clang::Expr *expr, clang::QualType to, CastKind kind)
        {
            auto loc = builder.getLocation(expr->getSourceRange());
            auto rty = types.convert(to);
            return make_value< Cast >( loc, rty, Visit(expr), kind );
        }

        auto make_region_builder(clang::Stmt *stmt)
        {
            return [stmt, this] (auto &bld, auto) {
                if (stmt) {
                    Visit(stmt);
                }
                spliceTrailingScopeBlocks(*bld.getBlock()->getParent());
            };
        }

        auto make_cond_builder(clang::Stmt *stmt)
        {
            return [stmt, this] (auto &bld, auto loc) {
                Visit(stmt);
                auto &op = bld.getBlock()->back();
                assert(op.getNumResults() == 1);
                auto cond = op.getResult(0);
                bld.template create< CondYieldOp >(loc, cond);
            };
        }

        auto make_value_builder(clang::Stmt *stmt)
        {
            return [stmt, this] (auto &bld, auto loc) {
                Visit(stmt);
                auto &op = bld.getBlock()->back();
                assert(op.getNumResults() == 1);
                auto cond = op.getResult(0);
                bld.template create< ValueYieldOp >(loc, cond);
            };
        }

        auto make_yield_true()
        {
            return [this] (auto &bld, auto loc) {
                auto t = builder.true_value(loc);
                bld.template create< CondYieldOp >(loc, t);
            };
        }

        template< typename LiteralType >
        auto make_scalar_literal(LiteralType *lit)
        {
            auto type = types.convert(lit->getType());
            auto loc  = builder.getLocation(lit->getSourceRange());
            return builder.constant(loc, type, lit->getValue());
        }

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

        // Binary Operations

        ValueOrStmt VisitBinPtrMemD(clang::BinaryOperator *expr)
        {
            UNREACHABLE( "unsupported BinPtrMemD" );
        }

        ValueOrStmt VisitBinPtrMemI(clang::BinaryOperator *expr)
        {
            UNREACHABLE( "unsupported BinPtrMemI" );
        }

        ValueOrStmt VisitBinMul(clang::BinaryOperator *expr)
        {
            if (auto val = make_ibin< MulIOp >(expr); check(val))
                return val;
            UNREACHABLE( "unsupported BinMul" );
        }

        ValueOrStmt VisitBinDiv(clang::BinaryOperator *expr)
        {
            if (auto val = make_ibin< DivUOp, DivSOp >(expr); check(val))
                return val;
            UNREACHABLE( "unsupported BinDiv" );
        }

        ValueOrStmt VisitBinRem(clang::BinaryOperator *expr)
        {
            if (auto val = make_ibin< RemUOp, RemSOp >(expr); check(val))
                return val;
            UNREACHABLE( "unsupported BinRem" );
        }

        ValueOrStmt VisitBinAdd(clang::BinaryOperator *expr)
        {
            if (auto val = make_ibin< AddIOp >(expr); check(val))
                return val;
            UNREACHABLE( "unsupported addition type" );
        }

        ValueOrStmt VisitBinSub(clang::BinaryOperator *expr)
        {
            if (auto val = make_ibin< SubIOp >(expr); check(val))
                return val;
            UNREACHABLE( "unsupported BinSub" );
        }

        ValueOrStmt VisitBinShl(clang::BinaryOperator *expr)
        {
            UNREACHABLE( "unsupported BinShl" );
        }

        ValueOrStmt VisitBinShr(clang::BinaryOperator *expr)
        {
            UNREACHABLE( "unsupported BinShr" );
        }

        ValueOrStmt VisitBinLT(clang::BinaryOperator *expr)
        {
            if (auto val = make_icmp< Predicate::ult, Predicate::slt >(expr))
                return val;
            UNREACHABLE( "unsupported BinLT" );
        }

        ValueOrStmt VisitBinGT(clang::BinaryOperator *expr)
        {
            if (auto val = make_icmp< Predicate::ugt, Predicate::sgt >(expr))
                return val;
            UNREACHABLE( "unsupported BinGT" );
        }

        ValueOrStmt VisitBinLE(clang::BinaryOperator *expr)
        {
            if (auto val = make_icmp< Predicate::ule, Predicate::sle >(expr))
                return val;
            UNREACHABLE( "unsupported BinLE" );
        }

        ValueOrStmt VisitBinGE(clang::BinaryOperator *expr)
        {
            if (auto val = make_icmp< Predicate::uge, Predicate::sge >(expr))
                return val;
            UNREACHABLE( "unsupported BinGE" );
        }

        ValueOrStmt VisitBinEQ(clang::BinaryOperator *expr)
        {
            if (auto val = make_icmp< Predicate::eq >(expr))
                return val;
            UNREACHABLE( "unsupported BinEQ" );
        }

        ValueOrStmt VisitBinNE(clang::BinaryOperator *expr)
        {
            if (auto val = make_icmp< Predicate::ne >(expr))
                return val;
            UNREACHABLE( "unsupported BinNE" );
        }

        ValueOrStmt VisitBinAnd(clang::BinaryOperator *expr)
        {
            UNREACHABLE( "unsupported BinAnd" );
        }

        ValueOrStmt VisitBinXor(clang::BinaryOperator *expr)
        {
            UNREACHABLE( "unsupported BinXor" );
        }

        ValueOrStmt VisitBinOr(clang::BinaryOperator *expr)
        {
            UNREACHABLE( "unsupported BinOr" );
        }

        ValueOrStmt VisitBinLAnd(clang::BinaryOperator *expr)
        {
            UNREACHABLE( "unsupported BinLAnd" );
        }

        ValueOrStmt VisitBinLOr(clang::BinaryOperator *expr)
        {
            UNREACHABLE( "unsupported BinLOr" );
        }

        ValueOrStmt VisitBinAssign(clang::BinaryOperator *expr)
        {
            return make_bin< AssignOp >(expr);
        }

        // Compound Assignment Operations

        ValueOrStmt VisitBinMulAssign(clang::CompoundAssignOperator *expr)
        {
            if (auto val = make_ibin< MulIAssignOp >(expr); check(val))
                return val;
            UNREACHABLE( "unsupported BinMulAssign" );
        }

        ValueOrStmt VisitBinDivAssign(clang::CompoundAssignOperator *expr)
        {
            if (auto val = make_ibin< DivUAssignOp, DivSAssignOp >(expr); check(val))
                return val;
            UNREACHABLE( "unsupported BinDivAssign" );
        }

        ValueOrStmt VisitBinRemAssign(clang::CompoundAssignOperator *expr)
        {
            if (auto val = make_ibin< RemUAssignOp, RemSAssignOp >(expr); check(val))
                return val;
            UNREACHABLE( "unsupported BinRemAssign" );
        }

        ValueOrStmt VisitBinAddAssign(clang::CompoundAssignOperator *expr)
        {
            if (auto val = make_ibin< AddIAssignOp >(expr); check(val))
                return val;
            UNREACHABLE( "unsupported BinAddAssign" );
        }

        ValueOrStmt VisitBinSubAssign(clang::CompoundAssignOperator *expr)
        {
            if (auto val = make_ibin< SubIAssignOp >(expr); check(val))
                return val;
            UNREACHABLE( "unsupported BinSubAssign" );
        }

        ValueOrStmt VisitBinShlAssign(clang::CompoundAssignOperator *expr)
        {
            UNREACHABLE( "unsupported BinShlAssign" );
        }

        ValueOrStmt VisitBinShrAssign(clang::CompoundAssignOperator *expr)
        {
            UNREACHABLE( "unsupported BinShrAssign" );
        }

        ValueOrStmt VisitBinAndAssign(clang::CompoundAssignOperator *expr)
        {
            UNREACHABLE( "unsupported BinAndAssign" );
        }

        ValueOrStmt VisitBinOrAssign(clang::CompoundAssignOperator *expr)
        {
            UNREACHABLE( "unsupported BinOrAssign" );
        }

        ValueOrStmt VisitBinXorAssign(clang::CompoundAssignOperator *expr)
        {
            UNREACHABLE( "unsupported BinXorAssign" );
        }

        ValueOrStmt VisitBinComma(clang::BinaryOperator *expr)
        {
            UNREACHABLE( "unsupported BinComma" );
        }

        // Unary Operations

        ValueOrStmt VisitUnaryPostInc(clang::UnaryOperator *expr)
        {
            return make_inplace_unary< PostIncOp >(expr);
        }

        ValueOrStmt VisitUnaryPostDec(clang::UnaryOperator *expr)
        {
            return make_inplace_unary< PostDecOp >(expr);
        }

        ValueOrStmt VisitUnaryPreInc(clang::UnaryOperator *expr)
        {
            return make_inplace_unary< PreIncOp >(expr);
        }

        ValueOrStmt VisitUnaryPreDec(clang::UnaryOperator *expr)
        {
            return make_inplace_unary< PreDecOp >(expr);
        }

        ValueOrStmt VisitUnaryAddrOf(clang::UnaryOperator *expr)
        {
            return make_unary< AddressOf >(expr);
        }

        ValueOrStmt VisitUnaryDeref(clang::UnaryOperator *expr)
        {
            return make_unary< Deref >(expr);
        }

        ValueOrStmt VisitUnaryPlus(clang::UnaryOperator *expr)
        {
            return make_type_preserving_unary< PlusOp >(expr);
        }

        ValueOrStmt VisitUnaryMinus(clang::UnaryOperator *expr)
        {
            return make_type_preserving_unary< MinusOp >(expr);
        }

        ValueOrStmt VisitUnaryNot(clang::UnaryOperator *expr)
        {
            return make_type_preserving_unary< NotOp >(expr);
        }

        ValueOrStmt VisitUnaryLNot(clang::UnaryOperator *expr)
        {
            return make_type_preserving_unary< LNotOp >(expr);
        }

        ValueOrStmt VisitUnaryReal(clang::UnaryOperator *expr)
        {
            UNREACHABLE( "unsupported UnaryReal" );
        }

        ValueOrStmt VisitUnaryImag(clang::UnaryOperator *expr)
        {
            UNREACHABLE( "unsupported UnaryImag" );
        }

        ValueOrStmt VisitUnaryExtension(clang::UnaryOperator *expr)
        {
            UNREACHABLE( "unsupported UnaryExtension" );
        }

        ValueOrStmt VisitUnaryCoawait(clang::UnaryOperator *expr)
        {
            UNREACHABLE( "unsupported UnaryCoawait" );
        }

        // Assembky Statements

        ValueOrStmt VisitAsmStmt(clang::AsmStmt *stmt)
        {
            UNREACHABLE( "unsupported AsmStmt" );
        }

        ValueOrStmt VisitGCCAsmStmt(clang::GCCAsmStmt *stmt)
        {
            UNREACHABLE( "unsupported GCCAsmStmt" );
        }

        ValueOrStmt VisitMSAsmStmt(clang::MSAsmStmt *stmt)
        {
            UNREACHABLE( "unsupported MSAsmStmt" );
        }

        ValueOrStmt VisitCoroutineBodyStmt(clang::CoroutineBodyStmt *stmt)
        {
            UNREACHABLE( "unsupported CoroutineBodyStmt" );
        }

        ValueOrStmt VisitCoreturnStmt(clang::CoreturnStmt *stmt)
        {
            UNREACHABLE( "unsupported CoreturnStmt" );
        }

        ValueOrStmt VisitCoroutineSuspendExpr(clang::CoroutineSuspendExpr *expr)
        {
            UNREACHABLE( "unsupported CoroutineSuspendExpr" );
        }

        ValueOrStmt VisitCoawaitExpr(clang::CoawaitExpr *expr)
        {
            UNREACHABLE( "unsupported CoawaitExpr" );
        }

        ValueOrStmt VisitCoyieldExpr(clang::CoyieldExpr *expr)
        {
            UNREACHABLE( "unsupported CoyieldExpr" );
        }

        ValueOrStmt VisitDependentCoawaitExpr(clang::DependentCoawaitExpr *expr)
        {
            UNREACHABLE( "unsupported DependentCoawaitExpr" );
        }

        ValueOrStmt VisitAttributedStmt(clang::AttributedStmt *stmt)
        {
            UNREACHABLE( "unsupported AttributedStmt" );
        }

        // Statements

        ValueOrStmt VisitBreakStmt(clang::BreakStmt *stmt)
        {
            return make< BreakOp >(builder.getLocation(stmt->getSourceRange()));
        }

        ValueOrStmt VisitCXXCatchStmt(clang::CXXCatchStmt *stmt)
        {
            UNREACHABLE( "unsupported CXXCatchStmt" );
        }

        ValueOrStmt VisitCXXForRangeStmt(clang::CXXForRangeStmt *stmt)
        {
            UNREACHABLE( "unsupported CXXForRangeStmt" );
        }

        ValueOrStmt VisitCXXTryStmt(clang::CXXTryStmt *stmt)
        {
            UNREACHABLE( "unsupported CXXTryStmt" );
        }

        ValueOrStmt VisitCapturedStmt(clang::CapturedStmt *stmt)
        {
            UNREACHABLE( "unsupported CapturedStmt" );
        }

        ValueOrStmt VisitCompoundStmt(clang::CompoundStmt *stmt)
        {
            ScopedInsertPoint builder_scope(builder);

            auto loc = builder.getLocation(stmt->getSourceRange());

            ScopeOp scope = make< ScopeOp >(loc);
            auto &body = scope.body();
            body.push_back( new mlir::Block() );
            builder.setInsertionPointToStart( &body.front() );

            for (auto s : stmt->body()) {
                Visit(s);
            }

            return scope;
        }

        ValueOrStmt VisitContinueStmt(clang::ContinueStmt *stmt)
        {
            return make< ContinueOp >(builder.getLocation(stmt->getSourceRange()));
        }

        ValueOrStmt VisitDeclStmt(clang::DeclStmt *stmt)
        {
            assert(stmt->isSingleDecl());
            return Visit( *(stmt->decls().begin()) );
        }

        ValueOrStmt VisitDoStmt(clang::DoStmt *stmt)
        {
            auto loc = builder.getLocation(stmt->getSourceRange());
            auto cond_builder = make_cond_builder(stmt->getCond());
            auto body_builder = make_region_builder(stmt->getBody());
            return make< DoOp >(loc, body_builder, cond_builder);
        }

        // Expressions

        ValueOrStmt VisitAbstractConditionalOperator(clang::AbstractConditionalOperator *stmt)
        {
            UNREACHABLE( "unsupported AbstractConditionalOperator" );
        }

        ValueOrStmt VisitBinaryConditionalOperator(clang::BinaryConditionalOperator *stmt)
        {
            UNREACHABLE( "unsupported BinaryConditionalOperator" );
        }

        ValueOrStmt VisitConditionalOperator(clang::ConditionalOperator *stmt)
        {
            UNREACHABLE( "unsupported ConditionalOperator" );
        }

        ValueOrStmt VisitAddrLabelExpr(clang::AddrLabelExpr *expr)
        {
            UNREACHABLE( "unsupported AddrLabelExpr" );
        }

        ValueOrStmt VisitConstantExpr(clang::ConstantExpr *expr)
        {
            auto loc = builder.getLocation(expr->getSourceRange());
            auto type = types.convert(expr->getType());
            return builder.constant(loc, type, expr->getResultAsAPSInt() );
        }

        ValueOrStmt VisitArraySubscriptExpr(clang::ArraySubscriptExpr *expr)
        {
            UNREACHABLE( "unsupported ArraySubscriptExpr" );
        }

        ValueOrStmt VisitArrayTypeTraitExpr(clang::ArrayTypeTraitExpr *expr)
        {
            UNREACHABLE( "unsupported ArrayTypeTraitExpr" );
        }

        ValueOrStmt VisitAsTypeExpr(clang::AsTypeExpr *expr)
        {
            UNREACHABLE( "unsupported AsTypeExpr" );
        }

        ValueOrStmt VisitAtomicExpr(clang::AtomicExpr *expr)
        {
            UNREACHABLE( "unsupported AtomicExpr" );
        }

        ValueOrStmt VisitBlockExpr(clang::BlockExpr *expr)
        {
            UNREACHABLE( "unsupported BlockExpr" );
        }

        ValueOrStmt VisitCXXBindTemporaryExpr(clang::CXXBindTemporaryExpr *expr)
        {
            UNREACHABLE( "unsupported CXXBindTemporaryExpr" );
        }

        ValueOrStmt VisitCXXBoolLiteralExpr(const clang::CXXBoolLiteralExpr *lit)
        {
            return make_scalar_literal(lit);
        }

        ValueOrStmt VisitCXXConstructExpr(clang::CXXConstructExpr *expr)
        {
            UNREACHABLE( "unsupported CXXConstructExpr" );
        }

        ValueOrStmt VisitCXXTemporaryObjectExpr(clang::CXXTemporaryObjectExpr *expr)
        {
            UNREACHABLE( "unsupported CXXTemporaryObjectExpr" );
        }

        ValueOrStmt VisitCXXDefaultArgExpr(clang::CXXDefaultArgExpr *expr)
        {
            UNREACHABLE( "unsupported CXXDefaultArgExpr" );
        }

        ValueOrStmt VisitCXXDefaultInitExpr(clang::CXXDefaultInitExpr *expr)
        {
            UNREACHABLE( "unsupported CXXDefaultInitExpr" );
        }

        ValueOrStmt VisitCXXDeleteExpr(clang::CXXDeleteExpr *expr)
        {
            UNREACHABLE( "unsupported CXXDeleteExpr" );
        }

        ValueOrStmt VisitCXXDependentScopeMemberExpr(clang::CXXDependentScopeMemberExpr *expr)
        {
            UNREACHABLE( "unsupported CXXDependentScopeMemberExpr" );
        }

        ValueOrStmt VisitCXXNewExpr(clang::CXXNewExpr *expr)
        {
            UNREACHABLE( "unsupported CXXNewExpr" );
        }

        ValueOrStmt VisitCXXNoexceptExpr(clang::CXXNoexceptExpr *expr)
        {
            UNREACHABLE( "unsupported CXXNoexceptExpr" );
        }

        ValueOrStmt VisitCXXNullPtrLiteralExpr(clang::CXXNullPtrLiteralExpr *expr)
        {
            UNREACHABLE( "unsupported CXXNullPtrLiteralExpr" );
        }

        ValueOrStmt VisitCXXPseudoDestructorExpr(clang::CXXPseudoDestructorExpr *expr)
        {
            UNREACHABLE( "unsupported CXXPseudoDestructorExpr" );
        }

        ValueOrStmt VisitCXXScalarValueInitExpr(clang::CXXScalarValueInitExpr *expr)
        {
            UNREACHABLE( "unsupported CXXScalarValueInitExpr" );
        }

        ValueOrStmt VisitCXXStdInitializerListExpr(clang::CXXStdInitializerListExpr *expr)
        {
            UNREACHABLE( "unsupported CXXStdInitializerListExpr" );
        }

        ValueOrStmt VisitCXXThisExpr(clang::CXXThisExpr *expr)
        {
            UNREACHABLE( "unsupported CXXThisExpr" );
        }

        ValueOrStmt VisitCXXThrowExpr(clang::CXXThrowExpr *expr)
        {
            UNREACHABLE( "unsupported CXXThrowExpr" );
        }

        ValueOrStmt VisitCXXTypeidExpr(clang::CXXTypeidExpr *expr)
        {
            UNREACHABLE( "unsupported CXXTypeidExpr" );
        }

        ValueOrStmt VisitCXXFoldExpr(clang::CXXFoldExpr *expr)
        {
            UNREACHABLE( "unsupported CXXFoldExpr" );
        }

        ValueOrStmt VisitCXXUnresolvedConstructExpr(clang::CXXUnresolvedConstructExpr *expr)
        {
            UNREACHABLE( "unsupported CXXUnresolvedConstructExpr" );
        }

        ValueOrStmt VisitCXXUuidofExpr(clang::CXXUuidofExpr *expr)
        {
            UNREACHABLE( "unsupported CXXUuidofExpr" );
        }

        mlir::FuncOp VisitDirectCallee(clang::FunctionDecl *callee)
        {
            auto name = callee->getName();
            auto fn = mod->lookupSymbol< mlir::FuncOp >( name );
            CHECK( fn, "missing function symbol {}", name );
            return fn;
        }

        mlir::Value VisitIndirectCallee(clang::Expr *callee)
        {
            return std::get< Value >( Visit(callee) );
        }

        using Arguments = llvm::SmallVector< Value, 2 >;

        Arguments VisitArguments(clang::CallExpr *expr)
        {
            Arguments args;
            for (const auto &arg : expr->arguments()) {
                args.push_back( std::get< Value >( Visit(arg) ) );
            }
            return args;
        }

        ValueOrStmt VisitDirectCall(clang::CallExpr *expr)
        {
            auto loc = builder.getLocation(expr->getSourceRange());
            auto callee = VisitDirectCallee(expr->getDirectCallee());
            auto args = VisitArguments(expr);
            return make_value< CallOp >( loc, callee, args );
        }

        ValueOrStmt VisitIndirectCall(clang::CallExpr *expr)
        {
            auto loc = builder.getLocation(expr->getSourceRange());
            auto callee = VisitIndirectCallee(expr->getCallee());
            auto args = VisitArguments(expr);
            return make_value< IndirectCallOp >( loc, callee, args );
        }

        ValueOrStmt VisitCallExpr(clang::CallExpr *expr)
        {
            if (expr->getDirectCallee()) {
                return VisitDirectCall(expr);
            }

            return VisitIndirectCall(expr);
        }

        ValueOrStmt VisitCUDAKernelCallExpr(clang::CUDAKernelCallExpr *expr)
        {
            UNREACHABLE( "unsupported CUDAKernelCallExpr" );
        }

        ValueOrStmt VisitCXXMemberCallExpr(clang::CXXMemberCallExpr *expr)
        {
            UNREACHABLE( "unsupported CXXMemberCallExpr" );
        }

        ValueOrStmt VisitCXXOperatorCallExpr(clang::CXXOperatorCallExpr *expr)
        {
            UNREACHABLE( "unsupported CXXOperatorCallExpr" );
        }

        ValueOrStmt VisitUserDefinedLiteral(clang::UserDefinedLiteral *lit)
        {
            UNREACHABLE( "unsupported UserDefinedLiteral" );
        }

        ValueOrStmt VisitCStyleCastExpr(clang::CStyleCastExpr *expr)
        {
            return make_cast< CStyleCastOp >( expr->getSubExpr(), expr->getType(), cast_kind(expr) );
        }

        ValueOrStmt VisitCXXFunctionalCastExpr(clang::CXXFunctionalCastExpr *expr)
        {
            UNREACHABLE( "unsupported CXXFunctionalCastExpr" );
        }

        ValueOrStmt VisitCXXConstCastExpr(clang::CXXConstCastExpr *expr)
        {
            UNREACHABLE( "unsupported CXXConstCastExpr" );
        }

        ValueOrStmt VisitCXXDynamicCastExpr(clang::CXXDynamicCastExpr *expr)
        {
            UNREACHABLE( "unsupported CXXDynamicCastExpr" );
        }

        ValueOrStmt VisitCXXReinterpretCastExpr(clang::CXXReinterpretCastExpr *expr)
        {
            UNREACHABLE( "unsupported CXXReinterpretCastExpr" );
        }

        ValueOrStmt VisitCXXStaticCastExpr(clang::CXXStaticCastExpr *expr)
        {
            UNREACHABLE( "unsupported CXXStaticCastExpr" );
        }

        ValueOrStmt VisitObjCBridgedCastExpr(clang::ObjCBridgedCastExpr *expr)
        {
            UNREACHABLE( "unsupported ObjCBridgedCastExpr" );
        }

        ValueOrStmt VisitImplicitCastExpr(clang::ImplicitCastExpr *expr)
        {
            return make_cast< ImplicitCastOp >( expr->getSubExpr(), expr->getType(), cast_kind(expr) );
        }

        ValueOrStmt VisitCharacterLiteral(clang::CharacterLiteral *lit)
        {
            return make_scalar_literal(lit);
        }

        ValueOrStmt VisitChooseExpr(clang::ChooseExpr *expr)
        {
            UNREACHABLE( "unsupported ChooseExpr" );
        }

        ValueOrStmt VisitCompoundLiteralExpr(clang::CompoundLiteralExpr *expr)
        {
            UNREACHABLE( "unsupported CompoundLiteralExpr" );
        }

        ValueOrStmt VisitConvertVectorExpr(clang::ConvertVectorExpr *expr)
        {
            UNREACHABLE( "unsupported ConvertVectorExpr" );
        }

        ValueOrStmt VisitDeclRefExpr(clang::DeclRefExpr *expr)
        {
            auto loc = builder.getLocation(expr->getSourceRange());

            // TODO(Heno): deal with function declaration

            // TODO(Heno): deal with enum constant declaration

            auto named = expr->getDecl()->getUnderlyingDecl();
            auto rty = types.convert(expr->getType());
            return make_value< DeclRefOp >( loc, rty, named->getNameAsString() );
        }

        ValueOrStmt VisitDependentScopeDeclRefExpr(clang::DependentScopeDeclRefExpr *expr)
        {
            UNREACHABLE( "unsupported DependentScopeDeclRefExpr" );
        }

        ValueOrStmt VisitDesignatedInitExpr(clang::DesignatedInitExpr *expr)
        {
            UNREACHABLE( "unsupported DesignatedInitExpr" );
        }

        ValueOrStmt VisitExprWithCleanups(clang::ExprWithCleanups *expr)
        {
            UNREACHABLE( "unsupported ExprWithCleanups" );
        }

        ValueOrStmt VisitExpressionTraitExpr(clang::ExpressionTraitExpr *expr)
        {
            UNREACHABLE( "unsupported ExpressionTraitExpr" );
        }

        ValueOrStmt VisitExtVectorElementExpr(clang::ExtVectorElementExpr *expr)
        {
            UNREACHABLE( "unsupported ExtVectorElementExpr" );
        }

        ValueOrStmt VisitFloatingLiteral(clang::FloatingLiteral *lit)
        {
            return make_scalar_literal(lit);
        }

        ValueOrStmt VisitFunctionParmPackExpr(clang::FunctionParmPackExpr *expr)
        {
            UNREACHABLE( "unsupported FunctionParmPackExpr" );
        }

        ValueOrStmt VisitGNUNullExpr(clang::GNUNullExpr *expr)
        {
            UNREACHABLE( "unsupported GNUNullExpr" );
        }

        ValueOrStmt VisitGenericSelectionExpr(clang::GenericSelectionExpr *expr)
        {
            UNREACHABLE( "unsupported GenericSelectionExpr" );
        }

        ValueOrStmt VisitImaginaryLiteral(clang::ImaginaryLiteral *lit)
        {
            UNREACHABLE( "unsupported ImaginaryLiteral" );
        }

        ValueOrStmt VisitFixedPointLiteral(clang::FixedPointLiteral *lit)
        {
            UNREACHABLE( "unsupported FixedPointLiteral" );
        }

        ValueOrStmt VisitImplicitValueInitExpr(clang::ImplicitValueInitExpr *expr)
        {
            UNREACHABLE( "unsupported ImplicitValueInitExpr" );
        }

        ValueOrStmt VisitInitListExpr(clang::InitListExpr *expr)
        {
            UNREACHABLE( "unsupported InitListExpr" );
        }

        ValueOrStmt VisitIntegerLiteral(const clang::IntegerLiteral *lit)
        {
            return make_scalar_literal(lit);
        }

        ValueOrStmt VisitLambdaExpr(clang::LambdaExpr *expr)
        {
            UNREACHABLE( "unsupported LambdaExpr" );
        }

        ValueOrStmt VisitMSPropertyRefExpr(clang::MSPropertyRefExpr *expr)
        {
            UNREACHABLE( "unsupported MSPropertyRefExpr" );
        }

        ValueOrStmt VisitMaterializeTemporaryExpr(clang::MaterializeTemporaryExpr *expr)
        {
            UNREACHABLE( "unsupported MaterializeTemporaryExpr" );
        }

        ValueOrStmt VisitMemberExpr(clang::MemberExpr *expr)
        {
            UNREACHABLE( "unsupported MemberExpr" );
        }

        ValueOrStmt VisitObjCArrayLiteral(clang::ObjCArrayLiteral *expr)
        {
            UNREACHABLE( "unsupported ObjCArrayLiteral" );
        }

        ValueOrStmt VisitObjCBoolLiteralExpr(clang::ObjCBoolLiteralExpr *expr)
        {
            UNREACHABLE( "unsupported ObjCBoolLiteralExpr" );
        }

        ValueOrStmt VisitObjCBoxedExpr(clang::ObjCBoxedExpr *expr)
        {
            UNREACHABLE( "unsupported ObjCBoxedExpr" );
        }

        ValueOrStmt VisitObjCDictionaryLiteral(clang::ObjCDictionaryLiteral *lit)
        {
            UNREACHABLE( "unsupported ObjCDictionaryLiteral" );
        }

        ValueOrStmt VisitObjCEncodeExpr(clang::ObjCEncodeExpr *expr)
        {
            UNREACHABLE( "unsupported ObjCEncodeExpr" );
        }

        ValueOrStmt VisitObjCIndirectCopyRestoreExpr(clang::ObjCIndirectCopyRestoreExpr *expr)
        {
            UNREACHABLE( "unsupported ObjCIndirectCopyRestoreExpr" );
        }

        ValueOrStmt VisitObjCIsaExpr(clang::ObjCIsaExpr *expr)
        {
            UNREACHABLE( "unsupported ObjCIsaExpr" );
        }

        ValueOrStmt VisitObjCIvarRefExpr(clang::ObjCIvarRefExpr *expr)
        {
            UNREACHABLE( "unsupported ObjCIvarRefExpr" );
        }

        ValueOrStmt VisitObjCMessageExpr(clang::ObjCMessageExpr *expr)
        {
            UNREACHABLE( "unsupported ObjCMessageExpr" );
        }

        ValueOrStmt VisitObjCPropertyRefExpr(clang::ObjCPropertyRefExpr *expr)
        {
            UNREACHABLE( "unsupported ObjCPropertyRefExpr" );
        }

        ValueOrStmt VisitObjCProtocolExpr(clang::ObjCProtocolExpr *expr)
        {
            UNREACHABLE( "unsupported ObjCProtocolExpr" );
        }

        ValueOrStmt VisitObjCSelectorExpr(clang::ObjCSelectorExpr *expr)
        {
            UNREACHABLE( "unsupported ObjCSelectorExpr" );
        }

        ValueOrStmt VisitObjCStringLiteral(clang::ObjCStringLiteral *lit)
        {
            UNREACHABLE( "unsupported ObjCStringLiteral" );
        }

        ValueOrStmt VisitObjCSubscriptRefExpr(clang::ObjCSubscriptRefExpr *expr)
        {
            UNREACHABLE( "unsupported ObjCSubscriptRefExpr" );
        }

        ValueOrStmt VisitOffsetOfExpr(clang::OffsetOfExpr *expr)
        {
            UNREACHABLE( "unsupported OffsetOfExpr" );
        }

        ValueOrStmt VisitOpaqueValueExpr(clang::OpaqueValueExpr *expr)
        {
            UNREACHABLE( "unsupported OpaqueValueExpr" );
        }

        ValueOrStmt VisitOverloadExpr(clang::OverloadExpr *expr)
        {
            UNREACHABLE( "unsupported OverloadExpr" );
        }

        ValueOrStmt VisitUnresolvedLookupExpr(clang::UnresolvedLookupExpr *expr)
        {
            UNREACHABLE( "unsupported UnresolvedLookupExpr" );
        }

        ValueOrStmt VisitUnresolvedMemberExpr(clang::UnresolvedMemberExpr *expr)
        {
            UNREACHABLE( "unsupported UnresolvedMemberExpr" );
        }

        ValueOrStmt VisitPackExpansionExpr(clang::PackExpansionExpr *expr)
        {
            UNREACHABLE( "unsupported PackExpansionExpr" );
        }

        ValueOrStmt VisitParenExpr(clang::ParenExpr *expr)
        {
            UNREACHABLE( "unsupported ParenExpr" );
        }

        ValueOrStmt VisitParenListExpr(clang::ParenListExpr *expr)
        {
            UNREACHABLE( "unsupported ParenListExpr" );
        }

        ValueOrStmt VisitPredefinedExpr(clang::PredefinedExpr *expr)
        {
            UNREACHABLE( "unsupported PredefinedExpr" );
        }

        ValueOrStmt VisitPseudoObjectExpr(clang::PseudoObjectExpr *expr)
        {
            UNREACHABLE( "unsupported PseudoObjectExpr" );
        }

        ValueOrStmt VisitShuffleVectorExpr(clang::ShuffleVectorExpr *expr)
        {
            UNREACHABLE( "unsupported ShuffleVectorExpr" );
        }

        ValueOrStmt VisitSizeOfPackExpr(clang::SizeOfPackExpr *expr)
        {
            UNREACHABLE( "unsupported SizeOfPackExpr" );
        }

        ValueOrStmt VisitStmtExpr(clang::StmtExpr *expr)
        {
            UNREACHABLE( "unsupported StmtExpr" );
        }

        ValueOrStmt VisitStringLiteral(clang::StringLiteral *lit)
        {
            auto type = types.convert(lit->getType());
            auto loc  = builder.getLocation(lit->getSourceRange());
            return builder.constant(loc, type, lit->getString());
        }

        ValueOrStmt VisitSubstNonTypeTemplateParmExpr(clang::SubstNonTypeTemplateParmExpr *expr)
        {
            UNREACHABLE( "unsupported SubstNonTypeTemplateParmExpr" );
        }

        ValueOrStmt VisitSubstNonTypeTemplateParmPackExpr(clang::SubstNonTypeTemplateParmPackExpr *expr)
        {
            UNREACHABLE( "unsupported SubstNonTypeTemplateParmPackExpr" );
        }

        ValueOrStmt VisitTypeTraitExpr(clang::TypeTraitExpr *expr)
        {
            UNREACHABLE( "unsupported TypeTraitExpr" );
        }

        ValueOrStmt VisitUnaryExprOrTypeTraitExpr(clang::UnaryExprOrTypeTraitExpr *expr)
        {
            UNREACHABLE( "unsupported UnaryExprOrTypeTraitExpr" );
        }

        ValueOrStmt VisitSourceLocExpr(clang::SourceLocExpr *expr)
        {
            UNREACHABLE( "unsupported SourceLocExpr" );
        }

        ValueOrStmt VisitVAArgExpr(clang::VAArgExpr *expr)
        {
            UNREACHABLE( "unsupported VAArgExpr" );
        }

        // Statements

        ValueOrStmt VisitForStmt(clang::ForStmt *stmt)
        {
            auto loc = builder.getLocation(stmt->getSourceRange());

            auto init_builder = make_region_builder(stmt->getInit());
            auto incr_builder = make_region_builder(stmt->getInc());
            auto body_builder = make_region_builder(stmt->getBody());

            auto cond = stmt->getCond();

            if (cond) {
                auto cond_builder = make_cond_builder(cond);
                return make< ForOp >(loc, init_builder, cond_builder, incr_builder, body_builder);
            }
            return make< ForOp >(loc, init_builder, make_yield_true(), incr_builder, body_builder);
        }

        ValueOrStmt VisitGotoStmt(clang::GotoStmt *stmt)
        {
            UNREACHABLE( "unsupported GotoStmt" );
        }

        ValueOrStmt VisitIfStmt(clang::IfStmt *stmt)
        {
            auto loc = builder.getLocation(stmt->getSourceRange());

            auto cond_builder = make_cond_builder(stmt->getCond());
            auto then_builder = make_region_builder(stmt->getThen());

            if (stmt->getElse()) {
                return make< IfOp >(loc, cond_builder, then_builder, make_region_builder(stmt->getElse()));
            }
            return make< IfOp >(loc, cond_builder, then_builder);
        }

        ValueOrStmt VisitIndirectGotoStmt(clang::IndirectGotoStmt *stmt)
        {
            UNREACHABLE( "unsupported IndirectGotoStmt" );
        }

        ValueOrStmt VisitLabelStmt(clang::LabelStmt *stmt)
        {
            UNREACHABLE( "unsupported LabelStmt" );
        }

        ValueOrStmt VisitMSDependentExistsStmt(clang::MSDependentExistsStmt *stmt)
        {
            UNREACHABLE( "unsupported MSDependentExistsStmt" );
        }

        ValueOrStmt VisitNullStmt(clang::NullStmt *stmt)
        {
            UNREACHABLE( "unsupported NullStmt" );
        }

        ValueOrStmt VisitOMPBarrierDirective(clang::OMPBarrierDirective *dir)
        {
            UNREACHABLE( "unsupported OMPBarrierDirective" );
        }

        ValueOrStmt VisitOMPCriticalDirective(clang::OMPCriticalDirective *dir)
        {
            UNREACHABLE( "unsupported OMPCriticalDirective" );
        }

        ValueOrStmt VisitOMPFlushDirective(clang::OMPFlushDirective *dir)
        {
            UNREACHABLE( "unsupported OMPFlushDirective" );
        }

        ValueOrStmt VisitOMPForDirective(clang::OMPForDirective *dir)
        {
            UNREACHABLE( "unsupported OMPForDirective" );
        }

        ValueOrStmt VisitOMPMasterDirective(clang::OMPMasterDirective *dir)
        {
            UNREACHABLE( "unsupported OMPMasterDirective" );
        }

        ValueOrStmt VisitOMPParallelDirective(clang::OMPParallelDirective *dir)
        {
            UNREACHABLE( "unsupported OMPParallelDirective" );
        }

        ValueOrStmt VisitOMPParallelForDirective(clang::OMPParallelForDirective *dir)
        {
            UNREACHABLE( "unsupported OMPParallelForDirective" );
        }

        ValueOrStmt VisitOMPParallelSectionsDirective(clang::OMPParallelSectionsDirective *dir)
        {
            UNREACHABLE( "unsupported OMPParallelSectionsDirective" );
        }

        ValueOrStmt VisitOMPSectionDirective(clang::OMPSectionDirective *dir)
        {
            UNREACHABLE( "unsupported OMPSectionDirective" );
        }

        ValueOrStmt VisitOMPSectionsDirective(clang::OMPSectionsDirective *dir)
        {
            UNREACHABLE( "unsupported OMPSectionsDirective" );
        }

        ValueOrStmt VisitOMPSimdDirective(clang::OMPSimdDirective *dir)
        {
            UNREACHABLE( "unsupported OMPSimdDirective" );
        }

        ValueOrStmt VisitOMPSingleDirective(clang::OMPSingleDirective *dir)
        {
            UNREACHABLE( "unsupported OMPSingleDirective" );
        }

        ValueOrStmt VisitOMPTaskDirective(clang::OMPTaskDirective *dir)
        {
            UNREACHABLE( "unsupported OMPTaskDirective" );
        }

        ValueOrStmt VisitOMPTaskwaitDirective(clang::OMPTaskwaitDirective *dir)
        {
            UNREACHABLE( "unsupported OMPTaskwaitDirective" );
        }

        ValueOrStmt VisitOMPTaskyieldDirective(clang::OMPTaskyieldDirective *dir)
        {
            UNREACHABLE( "unsupported OMPTaskyieldDirective" );
        }

        ValueOrStmt VisitObjCAtCatchStmt(clang::ObjCAtCatchStmt *stmt)
        {
            UNREACHABLE( "unsupported ObjCAtCatchStmt" );
        }

        ValueOrStmt VisitObjCAtFinallyStmt(clang::ObjCAtFinallyStmt *stmt)
        {
            UNREACHABLE( "unsupported ObjCAtFinallyStmt" );
        }

        ValueOrStmt VisitObjCAtSynchronizedStmt(clang::ObjCAtSynchronizedStmt *stmt)
        {
            UNREACHABLE( "unsupported ObjCAtSynchronizedStmt" );
        }

        ValueOrStmt VisitObjCAtThrowStmt(clang::ObjCAtThrowStmt *stmt)
        {
            UNREACHABLE( "unsupported ObjCAtThrowStmt" );
        }

        ValueOrStmt VisitObjCAtTryStmt(clang::ObjCAtTryStmt *stmt)
        {
            UNREACHABLE( "unsupported ObjCAtTryStmt" );
        }

        ValueOrStmt VisitObjCAutoreleasePoolStmt(clang::ObjCAutoreleasePoolStmt *stmt)
        {
            UNREACHABLE( "unsupported ObjCAutoreleasePoolStmt" );
        }

        ValueOrStmt VisitObjCForCollectionStmt(clang::ObjCForCollectionStmt *stmt)
        {
            UNREACHABLE( "unsupported ObjCForCollectionStmt" );
        }

        ValueOrStmt VisitReturnStmt(clang::ReturnStmt *stmt)
        {
            auto loc = builder.getLocation(stmt->getSourceRange());
            if ( auto ret = stmt->getRetValue() )
                return make< ReturnOp >(loc, Visit(ret));
            return make< ReturnOp >(loc);
        }

        ValueOrStmt VisitSEHExceptStmt(clang::SEHExceptStmt *stmt)
        {
            UNREACHABLE( "unsupported SEHExceptStmt" );
        }

        ValueOrStmt VisitSEHFinallyStmt(clang::SEHFinallyStmt *stmt)
        {
            UNREACHABLE( "unsupported SEHFinallyStmt" );
        }

        ValueOrStmt VisitSEHLeaveStmt(clang::SEHLeaveStmt *stmt)
        {
            UNREACHABLE( "unsupported SEHLeaveStmt" );
        }

        ValueOrStmt VisitSEHTryStmt(clang::SEHTryStmt *stmt)
        {
            UNREACHABLE( "unsupported SEHTryStmt" );
        }

        ValueOrStmt VisitCaseStmt(clang::CaseStmt *stmt)
        {
            auto loc = builder.getLocation(stmt->getSourceRange());
            auto lhs_builder = make_value_builder(stmt->getLHS());
            auto body_builder = make_region_builder(stmt->getSubStmt());
            return make< CaseOp >(loc, lhs_builder, body_builder);
        }

        ValueOrStmt VisitDefaultStmt(clang::DefaultStmt *stmt)
        {
            auto loc = builder.getLocation(stmt->getSourceRange());
            auto body_builder = make_region_builder(stmt->getSubStmt());
            return make< DefaultOp >(loc, body_builder);
        }

        ValueOrStmt VisitSwitchStmt(clang::SwitchStmt *stmt)
        {
            auto loc = builder.getLocation(stmt->getSourceRange());
            auto cond_builder = make_value_builder(stmt->getCond());
            auto body_builder = make_region_builder(stmt->getBody());
            if (stmt->getInit()) {
                return make< SwitchOp >(loc, make_region_builder(stmt->getInit()), cond_builder, body_builder);
            }
            return make< SwitchOp >(loc, nullptr, cond_builder, body_builder);
        }

        ValueOrStmt VisitWhileStmt(clang::WhileStmt *stmt)
        {
            auto loc = builder.getLocation(stmt->getSourceRange());
            auto cond_builder = make_cond_builder(stmt->getCond());
            auto body_builder = make_region_builder(stmt->getBody());
            return make< WhileOp >(loc, cond_builder, body_builder);
        }

        ValueOrStmt VisitBuiltinBitCastExpr(clang::BuiltinBitCastExpr *expr)
        {
            return make_cast< BuiltinBitCastOp >( expr->getSubExpr(), expr->getType(), cast_kind(expr) );
        }

        // Declarations

        ValueOrStmt VisitImportDecl(clang::ImportDecl *decl)
        {
            UNREACHABLE( "unsupported ImportDecl" );
        }

        ValueOrStmt VisitEmptyDecl(clang::EmptyDecl *decl)
        {
            UNREACHABLE( "unsupported EmptyDecl" );
        }

        ValueOrStmt VisitAccessSpecDecl(clang::AccessSpecDecl *decl)
        {
            UNREACHABLE( "unsupported AccessSpecDecl" );
        }

        ValueOrStmt VisitCapturedDecl(clang::CapturedDecl *decl)
        {
            UNREACHABLE( "unsupported CapturedDecl" );
        }

        ValueOrStmt VisitClassScopeFunctionSpecializationDecl(clang::ClassScopeFunctionSpecializationDecl *decl)
        {
            UNREACHABLE( "unsupported ClassScopeFunctionSpecializationDecl" );
        }

        ValueOrStmt VisitExportDecl(clang::ExportDecl *decl)
        {
            UNREACHABLE( "unsupported ExportDecl" );
        }

        ValueOrStmt VisitExternCContextDecl(clang::ExternCContextDecl *decl)
        {
            UNREACHABLE( "unsupported ExternCContextDecl" );
        }

        ValueOrStmt VisitFileScopeAsmDecl(clang::FileScopeAsmDecl *decl)
        {
            UNREACHABLE( "unsupported FileScopeAsmDecl" );
        }

        ValueOrStmt VisitStaticAssertDecl(clang::StaticAssertDecl *decl)
        {
            UNREACHABLE( "unsupported StaticAssertDecl" );
        }

        ValueOrStmt VisitTranslationUnitDecl(clang::TranslationUnitDecl *decl)
        {
            UNREACHABLE( "unsupported TranslationUnitDecl" );
        }

        ValueOrStmt VisitBindingDecl(clang::BindingDecl *decl)
        {
            UNREACHABLE( "unsupported BindingDecl" );
        }

        // ValueOrStmt VisitNamespaceDecl(clang::NamespaceDecl *decl)
        // {
        //     UNREACHABLE( "unsupported NamespaceDecl" );
        // }

        ValueOrStmt VisitNamespaceAliasDecl(clang::NamespaceAliasDecl *decl)
        {
            UNREACHABLE( "unsupported NamespaceAliasDecl" );
        }

        // ValueOrStmt VisitTypedefNameDecl(clang::TypedefNameDecl *decl)
        // {
        //     UNREACHABLE( "unsupported TypedefNameDecl" );
        // }

        ValueOrStmt VisitTypedefDecl(clang::TypedefDecl *decl)
        {
            auto loc = getLocation(decl->getSourceRange());
            auto name = decl->getName();

            auto type = [this, underlying = decl->getUnderlyingType()] () -> mlir::Type {
                if (underlying->isFunctionProtoType())
                    return convert(clang::cast< clang::FunctionType >(underlying));
                return convert(underlying);
            } ();

            return make< TypeDef >(loc, name, type);
        }

        ValueOrStmt VisitTypeAliasDecl(clang::TypeAliasDecl *decl)
        {
            UNREACHABLE( "unsupported TypeAliasDecl" );
        }
        ValueOrStmt VisitTemplateDecl(clang::TemplateDecl *decl)
        {
            UNREACHABLE( "unsupported TemplateDecl" );
        }

        ValueOrStmt VisitTypeAliasTemplateDecl(clang::TypeAliasTemplateDecl *decl)
        {
            UNREACHABLE( "unsupported TypeAliasTemplateDecl" );
        }

        ValueOrStmt VisitLabelDecl(clang::LabelDecl *decl)
        {
            UNREACHABLE( "unsupported LabelDecl" );
        }

        ValueOrStmt VisitEnumDecl(clang::EnumDecl *decl)
        {
            UNREACHABLE( "unsupported EnumDecl" );
        }

        ValueOrStmt VisitRecordDecl(clang::RecordDecl *decl)
        {
            UNREACHABLE( "unsupported RecordDecl" );
        }

        ValueOrStmt VisitEnumConstantDecl(clang::EnumConstantDecl *decl)
        {
            UNREACHABLE( "unsupported EnumConstantDecl" );
        }

        ValueOrStmt VisitFunctionDecl(clang::FunctionDecl *decl)
        {
            auto name = decl->getName();

            if ( auto fn = mod->lookupSymbol< mlir::FuncOp >( name ) )
                return fn;

            ScopedInsertPoint builder_scope(builder);
            llvm::ScopedHashTableScope scope(symbols);

            auto loc  = getLocation(decl->getSourceRange());
            auto type = convert(decl->getFunctionType());
            assert( type );

            auto fn = make< mlir::FuncOp >(loc, name, type);

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
                Visit(decl->getBody());
            }

            spliceTrailingScopeBlocks(fn);

            auto &last_block = fn.getBlocks().back();
            auto &ops = last_block.getOperations();
            builder.setInsertionPointToEnd(&last_block);

            if (ops.empty() || !ops.back().hasTrait< mlir::OpTrait::IsTerminator >()) {
                auto beg_loc = getLocation(decl->getBeginLoc());
                auto end_loc = getLocation(decl->getEndLoc());
                if (decl->getReturnType()->isVoidType()) {
                    make< ReturnOp >(end_loc);
                } else {
                    if (decl->isMain()) {
                        // return zero if no return is present in main
                    auto zero = builder.constant(end_loc, type.getResult(0), apint(0));
                        make< ReturnOp >(end_loc, zero);
                    } else {
                        make< UnreachableOp >(beg_loc);
                    }
                }
            }

            return fn;
        }

        ValueOrStmt VisitCXXMethodDecl(clang::CXXMethodDecl *decl)
        {
            UNREACHABLE( "unsupported CXXMethodDecl" );
        }

        ValueOrStmt VisitCXXConstructorDecl(clang::CXXConstructorDecl *decl)
        {
            UNREACHABLE( "unsupported CXXConstructorDecl" );
        }

        ValueOrStmt VisitCXXDestructorDecl(clang::CXXDestructorDecl *decl)
        {
            UNREACHABLE( "unsupported CXXDestructorDecl" );
        }

        ValueOrStmt VisitCXXConversionDecl(clang::CXXConversionDecl *decl)
        {
            UNREACHABLE( "unsupported CXXConversionDecl" );
        }

        ValueOrStmt VisitCXXDeductionGuideDecl(clang::CXXDeductionGuideDecl *decl)
        {
            UNREACHABLE( "unsupported CXXDeductionGuideDecl" );
        }

        ValueOrStmt VisitMSPropertyDecl(clang::MSPropertyDecl *decl)
        {
            UNREACHABLE( "unsupported MSPropertyDecl" );
        }

        ValueOrStmt VisitMSGuidDecl(clang::MSGuidDecl *decl)
        {
            UNREACHABLE( "unsupported MSGuidDecl" );
        }

        ValueOrStmt VisitFieldDecl(clang::FieldDecl *decl)
        {
            UNREACHABLE( "unsupported FieldDecl" );
        }

        ValueOrStmt VisitIndirectFieldDecl(clang::IndirectFieldDecl *decl)
        {
            UNREACHABLE( "unsupported IndirectFieldDecl" );
        }

        ValueOrStmt VisitFriendDecl(clang::FriendDecl *decl)
        {
            UNREACHABLE( "unsupported FriendDecl" );
        }

        ValueOrStmt VisitFriendTemplateDecl(clang::FriendTemplateDecl *decl)
        {
            UNREACHABLE( "unsupported FriendTemplateDecl" );
        }

        ValueOrStmt VisitObjCAtDefsFieldDecl(clang::ObjCAtDefsFieldDecl *decl)
        {
            UNREACHABLE( "unsupported ObjCAtDefsFieldDecl" );
        }

        ValueOrStmt VisitObjCIvarDecl(clang::ObjCIvarDecl *decl)
        {
            UNREACHABLE( "unsupported ObjCIvarDecl" );
        }

        template< typename Var, typename... Args >
        Var make_var(clang::VarDecl *decl, Args &&... args)
        {
            return make< Var >(std::forward< Args >(args)...);
        }

        template< typename Var >
        Var set_storage_qualifiers(clang::VarDecl *decl, Var var)
        {
            switch (decl->getStorageClass()) {
                case clang::SC_None: break;
                case clang::SC_Static:   var.setStaticStorage(); break;
                case clang::SC_Extern:   var.setExternalStorage(); break;
                case clang::SC_Auto:     var.setAutoStorage(); break;
                case clang::SC_Register: var.setRegisterStorage(); break;
                default:
                    UNREACHABLE("unsupported storage type");
            }

            if (decl->getStorageDuration() == clang::SD_Thread) {
                var.setThreadLocalStorage();
            }

            return var;
        }

        template< typename... Args >
        ValueOrStmt make_vardecl(clang::VarDecl *decl, Args &&... args)
        {
            return [&] () -> Stmt {
                if (decl->isFileVarDecl()) {
                    auto value = make_var< GlobalOp >(decl, std::forward< Args >(args)...);
                    return set_storage_qualifiers(decl, value);
                } else {
                    auto value = make_var< VarOp >(decl, std::forward< Args >(args)...);
                    return set_storage_qualifiers(decl, value);
                }
            } ();
        }

        ValueOrStmt VisitVarDecl(clang::VarDecl *decl)
        {
            auto ty   = convert(decl->getType());
            auto name = decl->getUnderlyingDecl()->getName();
            auto loc  = getEndLocation(decl->getSourceRange());

            if (decl->getInit()) {
                auto init = make_value_builder(decl->getInit());
                return make_vardecl(decl, loc, ty, name, init);
            }

            return make_vardecl(decl, loc, ty, name);
        }

        ValueOrStmt VisitDecompositionDecl(clang::DecompositionDecl *decl)
        {
            UNREACHABLE( "unsupported DecompositionDecl" );
        }


        ValueOrStmt VisitImplicitParamDecl(clang::ImplicitParamDecl *decl)
        {
            UNREACHABLE( "unsupported ImplicitParamDecl" );
        }

        // ValueOrStmt VisitUnresolvedUsingIfExistsDecl(clang::UnresolvedUsingIfExistsDecl *decl)
        // {
        //     UNREACHABLE( "unsupported UnresolvedUsingIfExistsDecl" );
        // }

        ValueOrStmt VisitParmVarDecl(clang::ParmVarDecl *decl)
        {
            UNREACHABLE( "unsupported ParmVarDecl" );
        }

        ValueOrStmt VisitObjCMethodDecl(clang::ObjCMethodDecl *decl)
        {
            UNREACHABLE( "unsupported ObjCMethodDecl" );
        }

        ValueOrStmt VisitObjCTypeParamDecl(clang::ObjCTypeParamDecl *decl)
        {
            UNREACHABLE( "unsupported ObjCTypeParamDecl" );
        }

        ValueOrStmt VisitObjCProtocolDecl(clang::ObjCProtocolDecl *decl)
        {
            UNREACHABLE( "unsupported ObjCProtocolDecl" );
        }

        ValueOrStmt VisitLinkageSpecDecl(clang::LinkageSpecDecl *decl)
        {
            UNREACHABLE( "unsupported LinkageSpecDecl" );
        }

        ValueOrStmt VisitUsingDecl(clang::UsingDecl *decl)
        {
            UNREACHABLE( "unsupported UsingDecl" );
        }

        ValueOrStmt VisitUsingShadowDecl(clang::UsingShadowDecl *decl)
        {
            UNREACHABLE( "unsupported UsingShadowDecl" );
        }

        ValueOrStmt VisitUsingDirectiveDecl(clang::UsingDirectiveDecl *decl)
        {
            UNREACHABLE( "unsupported UsingDirectiveDecl" );
        }

        ValueOrStmt VisitUsingPackDecl(clang::UsingPackDecl *decl)
        {
            UNREACHABLE( "unsupported UsingPackDecl" );
        }

        // ValueOrStmt VisitUsingEnumDecl(clang::UsingEnumDecl *decl)
        // {
        //     UNREACHABLE( "unsupported UsingEnumDecl" );
        // }

        ValueOrStmt VisitUnresolvedUsingValueDecl(clang::UnresolvedUsingValueDecl *decl)
        {
            UNREACHABLE( "unsupported UnresolvedUsingValueDecl" );
        }

        ValueOrStmt VisitUnresolvedUsingTypenameDecl(clang::UnresolvedUsingTypenameDecl *decl)
        {
            UNREACHABLE( "unsupported UnresolvedUsingTypenameDecl" );
        }

        ValueOrStmt VisitBuiltinTemplateDecl(clang::BuiltinTemplateDecl *decl)
        {
            UNREACHABLE( "unsupported BuiltinTemplateDecl" );
        }

        ValueOrStmt VisitConceptDecl(clang::ConceptDecl *decl)
        {
            UNREACHABLE( "unsupported ConceptDecl" );
        }

        ValueOrStmt VisitRedeclarableTemplateDecl(clang::RedeclarableTemplateDecl *decl)
        {
            UNREACHABLE( "unsupported RedeclarableTemplateDecl" );
        }

        ValueOrStmt VisitLifetimeExtendedTemporaryDecl(clang::LifetimeExtendedTemporaryDecl *decl)
        {
            UNREACHABLE( "unsupported LifetimeExtendedTemporaryDecl" );
        }

        ValueOrStmt VisitPragmaCommentDecl(clang::PragmaCommentDecl *decl)
        {
            UNREACHABLE( "unsupported PragmaCommentDecl" );
        }

        ValueOrStmt VisitPragmaDetectMismatchDecl(clang::PragmaDetectMismatchDecl *decl)
        {
            UNREACHABLE( "unsupported PragmaDetectMismatchDecl" );
        }

        ValueOrStmt VisitRequiresExprBodyDecl(clang::RequiresExprBodyDecl *decl)
        {
            UNREACHABLE( "unsupported RequiresExprBodyDecl" );
        }

        ValueOrStmt VisitObjCCompatibleAliasDecl(clang::ObjCCompatibleAliasDecl *decl)
        {
            UNREACHABLE( "unsupported ObjCCompatibleAliasDecl" );
        }

        ValueOrStmt VisitObjCCategoryDecl(clang::ObjCCategoryDecl *decl)
        {
            UNREACHABLE( "unsupported ObjCCategoryDecl" );
        }

        ValueOrStmt VisitObjCImplDecl(clang::ObjCImplDecl *decl)
        {
            UNREACHABLE( "unsupported ObjCImplDecl" );
        }

        ValueOrStmt VisitObjCInterfaceDecl(clang::ObjCInterfaceDecl *decl)
        {
            UNREACHABLE( "unsupported ObjCInterfaceDecl" );
        }

        ValueOrStmt VisitObjCCategoryImplDecl(clang::ObjCCategoryImplDecl *decl)
        {
            UNREACHABLE( "unsupported ObjCCategoryImplDecl" );
        }

        ValueOrStmt VisitObjCImplementationDecl(clang::ObjCImplementationDecl *decl)
        {
            UNREACHABLE( "unsupported ObjCImplementationDecl" );
        }

        ValueOrStmt VisitObjCPropertyDecl(clang::ObjCPropertyDecl *decl)
        {
            UNREACHABLE( "unsupported ObjCPropertyDecl" );
        }

        ValueOrStmt VisitObjCPropertyImplDecl(clang::ObjCPropertyImplDecl *decl)
        {
            UNREACHABLE( "unsupported ObjCPropertyImplDecl" );
        }

        ValueOrStmt VisitTemplateParamObjectDecl(clang::TemplateParamObjectDecl *decl)
        {
            UNREACHABLE( "unsupported TemplateParamObjectDecl" );
        }

        ValueOrStmt VisitTemplateTypeParmDecl(clang::TemplateTypeParmDecl *decl)
        {
            UNREACHABLE( "unsupported TemplateTypeParmDecl" );
        }

        ValueOrStmt VisitNonTypeTemplateParmDecl(clang::NonTypeTemplateParmDecl *decl)
        {
            UNREACHABLE( "unsupported NonTypeTemplateParmDecl" );
        }

        ValueOrStmt VisitTemplateTemplateParmDecl(clang::TemplateTemplateParmDecl *decl)
        {
            UNREACHABLE( "unsupported TemplateTemplateParmDecl" );
        }

        ValueOrStmt VisitClassTemplateDecl(clang::ClassTemplateDecl *decl)
        {
            UNREACHABLE( "unsupported ClassTemplateDecl" );
        }

        ValueOrStmt VisitClassTemplatePartialSpecializationDecl(clang::ClassTemplatePartialSpecializationDecl *decl)
        {
            UNREACHABLE( "unsupported ClassTemplatePartialSpecializationDecl" );
        }

        ValueOrStmt VisitClassTemplateSpecializationDecl(clang::ClassTemplateSpecializationDecl *decl)
        {
            UNREACHABLE( "unsupported ClassTemplateSpecializationDecl" );
        }

        ValueOrStmt VisitVarTemplateDecl(clang::VarTemplateDecl *decl)
        {
            UNREACHABLE( "unsupported VarTemplateDecl" );
        }

        ValueOrStmt VisitVarTemplateSpecializationDecl(clang::VarTemplateSpecializationDecl *decl)
        {
            UNREACHABLE( "unsupported VarTemplateSpecializationDecl" );
        }

        ValueOrStmt VisitVarTemplatePartialSpecializationDecl(clang::VarTemplatePartialSpecializationDecl *decl)
        {
            UNREACHABLE( "unsupported VarTemplatePartialSpecializationDecl" );
        }

        ValueOrStmt VisitFunctionTemplateDecl(clang::FunctionTemplateDecl *decl)
        {
            UNREACHABLE( "unsupported FunctionTemplateDecl" );
        }

        ValueOrStmt VisitConstructorUsingShadowDecl(clang::ConstructorUsingShadowDecl *decl)
        {
            UNREACHABLE( "unsupported ConstructorUsingShadowDecl" );
        }

        ValueOrStmt VisitOMPAllocateDecl(clang::OMPAllocateDecl *decl)
        {
            UNREACHABLE( "unsupported OMPAllocateDecl" );
        }

        ValueOrStmt VisitOMPRequiresDecl(clang::OMPRequiresDecl *decl)
        {
            UNREACHABLE( "unsupported OMPRequiresDecl" );
        }

        ValueOrStmt VisitOMPThreadPrivateDecl(clang::OMPThreadPrivateDecl *decl)
        {
            UNREACHABLE( "unsupported OMPThreadPrivateDecl" );
        }

        ValueOrStmt VisitOMPCapturedExprDecl(clang::OMPCapturedExprDecl *decl)
        {
            UNREACHABLE( "unsupported OMPCapturedExprDecl" );
        }

        ValueOrStmt VisitOMPDeclareReductionDecl(clang::OMPDeclareReductionDecl *decl)
        {
            UNREACHABLE( "unsupported OMPDeclareReductionDecl" );
        }

        ValueOrStmt VisitOMPDeclareMapperDecl(clang::OMPDeclareMapperDecl *decl)
        {
            UNREACHABLE( "unsupported OMPDeclareMapperDecl" );
        }

    private:
        mlir::OwningModuleRef &mod;

        VastBuilder     builder;
        TypeConverter   types;

        // Declare a variable in the current scope, return success if the variable
        // wasn't declared yet.
        logical_result declare(llvm::StringRef var, mlir::Value value)
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
        llvm::ScopedHashTable<llvm::StringRef, mlir::Value> symbols;
    };

    struct VastCodeGen : clang::ASTConsumer
    {
        VastCodeGen(mlir::MLIRContext &mctx, mlir::OwningModuleRef &mod, clang::ASTContext &actx)
            : mctx(mctx), mod(mod), actx(actx)
        {}

        bool HandleTopLevelDecl(clang::DeclGroupRef) override
        {
            UNREACHABLE("not implemented");
        }

        void HandleTranslationUnit(clang::ASTContext&) override
        {
            auto tu = actx.getTranslationUnitDecl();

            VastCodeGenVisitor visitor(mctx, mod, actx);

            for (const auto &decl : tu->decls())
                visitor.Visit(decl);
        }

    private:

        mlir::MLIRContext     &mctx;
        mlir::OwningModuleRef &mod;
        clang::ASTContext     &actx;
    };

    static llvm::cl::list<std::string> compiler_args(
        "ccopts", llvm::cl::ZeroOrMore, llvm::cl::desc("Specify compiler options")
    );

    static mlir::OwningModuleRef from_source_parser(const llvm::MemoryBuffer *input, mlir::MLIRContext *ctx)
    {
        ctx->loadDialect< HighLevelDialect >();
        ctx->loadDialect< mlir::StandardOpsDialect >();
        ctx->loadDialect< mlir::scf::SCFDialect >();

        mlir::OwningModuleRef mod(
            mlir::ModuleOp::create(
                mlir::FileLineColLoc::get(ctx, input->getBufferIdentifier(), /* line */ 0, /* column */ 0)
            )
        );

        auto ast = clang::tooling::buildASTFromCodeWithArgs(input->getBuffer(), compiler_args);

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
