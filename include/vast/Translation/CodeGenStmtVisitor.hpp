// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <clang/AST/StmtVisitor.h>
VAST_UNRELAX_WARNINGS

#include "vast/Translation/CodeGenMeta.hpp"
#include "vast/Translation/CodeGenBuilder.hpp"
#include "vast/Translation/CodeGenVisitorBase.hpp"
#include "vast/Translation/CodeGenVisitorLens.hpp"

#include "vast/Dialect/HighLevel/HighLevelDialect.hpp"
#include "vast/Dialect/HighLevel/HighLevelOps.hpp"

namespace vast::hl {

    CastKind cast_kind(const clang::CastExpr *expr);

    template< typename Derived >
    struct CodeGenStmtVisitorMixin
        : clang::ConstStmtVisitor< CodeGenStmtVisitorMixin< Derived >, Operation* >
        , CodeGenVisitorLens< CodeGenStmtVisitorMixin< Derived >, Derived >
        , CodeGenBuilderMixin< CodeGenStmtVisitorMixin< Derived >, Derived >
    {
        using LensType = CodeGenVisitorLens< CodeGenStmtVisitorMixin< Derived >, Derived >;

        using LensType::derived;
        using LensType::context;
        using LensType::mcontext;

        using LensType::meta_location;

        using LensType::visit;
        using LensType::visit_as_lvalue_type;

        using Builder = CodeGenBuilderMixin< CodeGenStmtVisitorMixin< Derived >, Derived >;

        using Builder::op_builder;

        using Builder::constant;

        using Builder::set_insertion_point_to_start;

        using Builder::make_yield_true;
        using Builder::make_cond_builder;
        using Builder::make_value_builder;
        using Builder::make_region_builder;

        using Builder::make_value_yield_region;

        template< typename Op, typename... Args >
        auto make(Args &&...args) {
            return this->template create< Op >(std::forward< Args >(args)...);
        }

        Operation* VisitCompoundStmt(const clang::CompoundStmt *stmt) {
            return this->template make_scoped< HighLevelScope >(meta_location(stmt), [&] {
                for (auto s : stmt->body()) {
                    visit(s);
                }
            });
        }

        //
        // Binary Operations
        //

        template< typename Op >
        Operation* VisitBinOp(const clang::BinaryOperator *op) {
            auto lhs = visit(op->getLHS())->getResult(0);
            auto rhs = visit(op->getRHS())->getResult(0);
            return make< Op >(meta_location(op), lhs, rhs);
        }

        template< typename UOp, typename SOp >
        Operation* VisitIBinOp(const clang::BinaryOperator *op) {
            auto ty = op->getType();
            if (ty->isUnsignedIntegerType())
                return VisitBinOp< UOp >(op);
            if (ty->isIntegerType())
                return VisitBinOp< SOp >(op);
            return nullptr;
        }

        template< Predicate pred >
        Operation* VisitCmp(const clang::BinaryOperator *op) {
            auto lhs = visit(op->getLHS())->getResult(0);
            auto rhs = visit(op->getRHS())->getResult(0);
            auto res = visit(op->getType());
            return make< CmpOp >(meta_location(op), res, pred, lhs, rhs);
        }

        template< Predicate upred, Predicate spred >
        Operation* VisitICmp(const clang::BinaryOperator *op) {
            auto ty = op->getLHS()->getType();
            if (ty->isUnsignedIntegerType())
                return VisitCmp< upred >(op);
            if (ty->isIntegerType())
                return VisitCmp< spred >(op);
            return nullptr;
        }

        // Operation* VisitBinPtrMemD(clang::BinaryOperator *op);
        // Operation* VisitBinPtrMemI(clang::BinaryOperator *op);

        Operation* VisitBinMul(const clang::BinaryOperator *op) {
            return VisitBinOp< MulIOp >(op);
        }

        Operation* VisitBinDiv(const clang::BinaryOperator *op) {
            return VisitIBinOp< DivUOp, DivSOp >(op);
        }

        Operation* VisitBinRem(const clang::BinaryOperator *op) {
            return VisitIBinOp< RemUOp, RemSOp >(op);
        }

        Operation* VisitBinAdd(const clang::BinaryOperator *op) {
            return VisitBinOp< AddIOp >(op);
        }

        Operation* VisitBinSub(const clang::BinaryOperator *op) {
            return VisitBinOp< SubIOp >(op);
        }

        Operation* VisitBinShl(const clang::BinaryOperator *op) {
            return VisitBinOp< BinShlOp >(op);
        }

        Operation* VisitBinShr(const clang::BinaryOperator *op) {
            return VisitBinOp< BinShrOp >(op);
        }

        Operation* VisitBinLT(const clang::BinaryOperator *op) {
            return VisitICmp< Predicate::ult, Predicate::slt >(op);
        }

        Operation* VisitBinGT(const clang::BinaryOperator *op) {
            return VisitICmp< Predicate::ugt, Predicate::sgt >(op);
        }

        Operation* VisitBinLE(const clang::BinaryOperator *op) {
            return VisitICmp< Predicate::ule, Predicate::sle >(op);
        }

        Operation* VisitBinGE(const clang::BinaryOperator *op) {
            return VisitICmp< Predicate::uge, Predicate::sge >(op);
        }

        Operation* VisitBinEQ(const clang::BinaryOperator *op) {
            return VisitCmp< Predicate::eq >(op);
        }

        Operation* VisitBinNE(const clang::BinaryOperator *op) {
            return VisitCmp< Predicate::ne >(op);
        }

        Operation* VisitBinAnd(const clang::BinaryOperator *op) {
            return VisitBinOp< BinAndOp >(op);
        }

        Operation* VisitBinXor(const clang::BinaryOperator *op) {
            return VisitBinOp< BinXorOp >(op);
        }

        Operation* VisitBinOr(const clang::BinaryOperator *op) {
            return VisitBinOp< BinOrOp >(op);
        }

        Operation* VisitBinLAnd(const clang::BinaryOperator *op) {
            auto lhs = visit(op->getLHS())->getResult(0);
            auto rhs = visit(op->getRHS())->getResult(0);
            auto ty  = visit(op->getType());
            return make< BinLAndOp >(meta_location(op), ty, lhs, rhs);
        }

        Operation* VisitBinLOr(const clang::BinaryOperator *op) {
            auto lhs = visit(op->getLHS())->getResult(0);
            auto rhs = visit(op->getRHS())->getResult(0);
            auto ty  = visit(op->getType());
            return make< BinLOrOp >(meta_location(op), ty, lhs, rhs);
        }

        Operation* VisitBinAssign(const clang::BinaryOperator *op) {
            return VisitBinOp< AssignOp >(op);
        }

        //
        // Compound Assignment Operations
        //

        Operation* VisitBinMulAssign(const clang::CompoundAssignOperator *op) {
            return VisitBinOp< MulIAssignOp >(op);
        }

        Operation* VisitBinDivAssign(const clang::CompoundAssignOperator *op) {
            return VisitIBinOp< DivUAssignOp, DivSAssignOp >(op);
        }

        Operation* VisitBinRemAssign(const clang::CompoundAssignOperator *op) {
            return VisitIBinOp< RemUAssignOp, RemSAssignOp >(op);
        }

        Operation* VisitBinAddAssign(const clang::CompoundAssignOperator *op) {
            return VisitBinOp< AddIAssignOp >(op);
        }

        Operation* VisitBinSubAssign(const clang::CompoundAssignOperator *op) {
            return VisitBinOp< SubIAssignOp >(op);
        }

        Operation* VisitBinShlAssign(const clang::CompoundAssignOperator *op) {
            return VisitBinOp< BinShlAssignOp >(op);
        }

        Operation* VisitBinShrAssign(const clang::CompoundAssignOperator *op) {
            return VisitBinOp< BinShrAssignOp >(op);
        }

        Operation* VisitBinAndAssign(const clang::CompoundAssignOperator *op) {
            return VisitBinOp< BinAndAssignOp >(op);
        }

        Operation* VisitBinOrAssign(const clang::CompoundAssignOperator *op) {
            return VisitBinOp< BinOrAssignOp >(op);
        }

        Operation* VisitBinXorAssign(const clang::CompoundAssignOperator *op) {
            return VisitBinOp< BinXorAssignOp >(op);
        }

        Operation* VisitBinComma(const clang::BinaryOperator *op) {
            auto lhs = visit(op->getLHS())->getResult(0);
            auto rhs = visit(op->getRHS())->getResult(0);
            auto ty  = visit(op->getType());
            return make< BinComma >(meta_location(op), ty, lhs, rhs);
        }

        //
        // Unary Operations
        //

        template< typename Op >
        Operation*  VisitUnary(const clang::UnaryOperator *op, Type rty) {
            auto arg = visit(op->getSubExpr())->getResult(0);
            return make< Op >(meta_location(op), rty, arg);
        }

        template< typename Op >
        Operation* VisitUnderlyingTypePreservingUnary(const clang::UnaryOperator *op) {
            auto arg = visit(op->getSubExpr())->getResult(0);
            auto type = arg.getType();
            if (auto ltype = type.template dyn_cast< LValueType >()) {
                type = ltype.getElementType();
            }
            return make< Op >(meta_location(op), type, arg);
        }

        Operation* VisitUnaryPostInc(const clang::UnaryOperator *op) {
            return VisitUnderlyingTypePreservingUnary< PostIncOp >(op);
        }

        Operation* VisitUnaryPostDec(const clang::UnaryOperator *op) {
            return VisitUnderlyingTypePreservingUnary< PostDecOp >(op);
        }

        Operation* VisitUnaryPreInc(const clang::UnaryOperator *op) {
            return VisitUnderlyingTypePreservingUnary< PreIncOp >(op);
        }

        Operation* VisitUnaryPreDec(const clang::UnaryOperator *op) {
            return VisitUnderlyingTypePreservingUnary< PreDecOp >(op);
        }

        Operation* VisitUnaryAddrOf(const clang::UnaryOperator *op) {
            return VisitUnary< AddressOf >(op, visit(op->getType()));
        }

        Operation* VisitUnaryDeref(const clang::UnaryOperator *op) {
            return VisitUnary< Deref >(op, visit_as_lvalue_type(op->getType()));
        }

        Operation* VisitUnaryPlus(const clang::UnaryOperator *op) {
            return VisitUnderlyingTypePreservingUnary< PlusOp >(op);
        }

        Operation* VisitUnaryMinus(const clang::UnaryOperator *op) {
            return VisitUnderlyingTypePreservingUnary< MinusOp >(op);
        }

        Operation* VisitUnaryNot(const clang::UnaryOperator *op) {
            return VisitUnderlyingTypePreservingUnary< NotOp >(op);
        }

        Operation* VisitUnaryLNot(const clang::UnaryOperator *op) {
            return VisitUnderlyingTypePreservingUnary< LNotOp >(op);
        }

        // Operation* VisitUnaryReal(const clang::UnaryOperator *op)
        // Operation* VisitUnaryImag(const clang::UnaryOperator *op)
        // Operation* VisitUnaryExtension(const clang::UnaryOperator *op)
        // Operation* VisitUnaryCoawait(const clang::UnaryOperator *op)

        //
        // Assembly Statements
        //

        // Operation* VisitAsmStmt(const clang::AsmStmt *stmt)
        // Operation* VisitGCCAsmStmt(const clang::GCCAsmStmt *stmt)
        // Operation* VisVisitMSAsmStmtitAsmStmt(const clang::MSAsmStmt *stmt)

        //
        // Coroutine Statements
        //

        // Operation* VisitCoroutineBodyStmt(const clang::CoroutineBodyStmt *stmt)
        // Operation* VisitCoreturnStmt(const clang::CoreturnStmt *stmt)
        // Operation* VisitCoroutineSuspendExpr(const clang::CoroutineSuspendExpr *stmt)
        // Operation* VisitCoawaitExpr(const clang::CoawaitExpr *expr)
        // Operation* VisitCoyieldExpr(const clang::CoyieldExpr *expr)
        // Operation* VisitDependentCoawaitExpr(const clang::DependentCoawaitExpr *expr)

        // Operation* VisitAttributedStmt(const clang::AttributedStmt *stmt)

        //
        // Cast Operations
        //

        Type VisitCastReturnType(const clang::CastExpr *expr, Type from) {
            auto to_rvalue_cast     = [&] { return visit(expr->getType()); };
            auto lvalue_cast        = [&] { return visit_as_lvalue_type(expr->getType()); };
            auto non_lvalue_cast    = [&] { return visit(expr->getType()); };
            auto keep_category_cast = [&] {
                if (from.isa< LValueType >())
                    return lvalue_cast();
                return non_lvalue_cast();
            };

            switch (expr->getCastKind()) {
                // case clang::CastKind::CK_Dependent:
                case clang::CastKind::CK_BitCast:               return non_lvalue_cast();
                case clang::CastKind::CK_LValueBitCast:         return lvalue_cast();
                case clang::CastKind::CK_LValueToRValueBitCast: return to_rvalue_cast();
                case clang::CastKind::CK_LValueToRValue:        return to_rvalue_cast();
                case clang::CastKind::CK_NoOp:                  return from;

                case clang::CastKind::CK_BaseToDerived:          return lvalue_cast();
                case clang::CastKind::CK_DerivedToBase:          return lvalue_cast();
                case clang::CastKind::CK_UncheckedDerivedToBase: return lvalue_cast();
                case clang::CastKind::CK_Dynamic:                return lvalue_cast();
                case clang::CastKind::CK_ToUnion:                return lvalue_cast();

                case clang::CastKind::CK_NullToPointer:          return non_lvalue_cast();
                case clang::CastKind::CK_ArrayToPointerDecay:
                // case clang::CastKind::CK_FunctionToPointerDecay:     return;
                // case clang::CastKind::CK_NullToMemberPointer:        return;
                // case clang::CastKind::CK_BaseToDerivedMemberPointer: return;
                // case clang::CastKind::CK_DerivedToBaseMemberPointer: return;
                // case clang::CastKind::CK_MemberPointerToBoolean:     return;
                // case clang::CastKind::CK_ReinterpretMemberPointer:   return;
                // case clang::CastKind::CK_UserDefinedConversion:      return;
                // case clang::CastKind::CK_ConstructorConversion:      return;
                    return keep_category_cast();

                case clang::CastKind::CK_IntegralToPointer:
                case clang::CastKind::CK_PointerToIntegral:
                case clang::CastKind::CK_PointerToBoolean :
                    return keep_category_cast();

                case clang::CastKind::CK_ToVoid:
                    return keep_category_cast();

                // case clang::CastKind::CK_VectorSplat: return;

                case clang::CastKind::CK_IntegralCast:
                case clang::CastKind::CK_IntegralToBoolean:
                case clang::CastKind::CK_IntegralToFloating:
                case clang::CastKind::CK_FloatingToFixedPoint:
                case clang::CastKind::CK_FixedPointToFloating:
                case clang::CastKind::CK_FixedPointCast:
                case clang::CastKind::CK_FixedPointToIntegral:
                case clang::CastKind::CK_IntegralToFixedPoint:
                case clang::CastKind::CK_FixedPointToBoolean:
                case clang::CastKind::CK_FloatingToIntegral:
                case clang::CastKind::CK_FloatingToBoolean:
                case clang::CastKind::CK_BooleanToSignedIntegral:
                case clang::CastKind::CK_FloatingCast:
                    return keep_category_cast();

                // case clang::CastKind::CK_CPointerToObjCPointerCast:
                // case clang::CastKind::CK_BlockPointerToObjCPointerCast:
                // case clang::CastKind::CK_AnyPointerToBlockPointerCast:
                // case clang::CastKind::CK_ObjCObjectLValueCast:

                case clang::CastKind::CK_FloatingRealToComplex:
                case clang::CastKind::CK_FloatingComplexToReal:
                case clang::CastKind::CK_FloatingComplexToBoolean:
                case clang::CastKind::CK_FloatingComplexCast:
                case clang::CastKind::CK_FloatingComplexToIntegralComplex:
                case clang::CastKind::CK_IntegralRealToComplex:
                case clang::CastKind::CK_IntegralComplexToReal:
                case clang::CastKind::CK_IntegralComplexToBoolean:
                case clang::CastKind::CK_IntegralComplexCast:
                case clang::CastKind::CK_IntegralComplexToFloatingComplex:
                    return keep_category_cast();

                // case clang::CastKind::CK_ARCProduceObject:
                // case clang::CastKind::CK_ARCConsumeObject:
                // case clang::CastKind::CK_ARCReclaimReturnedObject:
                // case clang::CastKind::CK_ARCExtendBlockObject:

                // case clang::CastKind::CK_AtomicToNonAtomic:
                // case clang::CastKind::CK_NonAtomicToAtomic:

                // case clang::CastKind::CK_CopyAndAutoreleaseBlockObject:
                // case clang::CastKind::CK_BuiltinFnToFnPtr:

                // case clang::CastKind::CK_ZeroToOCLOpaqueType:
                // case clang::CastKind::CK_AddressSpaceConversion:
                // case clang::CastKind::CK_IntToOCLSampler:

                // case clang::CastKind::CK_MatrixCast:
                default:
                    VAST_UNREACHABLE( "unsupported cast kind" );
            }
        }


        template< typename Cast >
        Operation* VisitCast(const clang::CastExpr *expr) {
            auto arg = visit(expr->getSubExpr());
            auto rty = VisitCastReturnType(expr, arg->getResultTypes().front());
            return make< Cast >(meta_location(expr), rty, arg->getResult(0), cast_kind(expr));
        }

        Operation* VisitImplicitCastExpr(const clang::ImplicitCastExpr *expr) {
             return VisitCast< ImplicitCastOp >(expr);
        }

        Operation* VisitCStyleCastExpr(const clang::CStyleCastExpr *expr) {
            return VisitCast< CStyleCastOp >(expr);
        }

        Operation* VisitBuiltinBitCastExpr(const clang::BuiltinBitCastExpr *expr) {
            return VisitCast< BuiltinBitCastOp >(expr);
        }

        // Operation* VisitCXXFunctionalCastExpr(const clang::CXXFunctionalCastExpr *expr)
        // Operation* VisitCXXConstCastExpr(const clang::CXXConstCastExpr *expr)
        // Operation* VisitCXXDynamicCastExpr(const clang::CXXDynamicCastExpr *expr)
        // Operation* VisitCXXReinterpretCastExpr(const clang::CXXReinterpretCastExpr *expr)
        // Operation* VisitCXXStaticCastExpr(const clang::CXXStaticCastExpr *expr)

        //
        // Other Expressions
        //

        Operation* VisitDeclStmt(const clang::DeclStmt *stmt) {
            // TODO maybe scoped?
            Operation * last = nullptr;
            for (auto decl : stmt->decls()) {
                last = visit(decl);
            }
            return last;
        }

        Type getLValueReturnType(const clang::Expr *expr) {
            return visit_as_lvalue_type(expr->getType());
        }

        const clang::VarDecl * getDeclForVarRef(const clang::DeclRefExpr *expr) {
            return clang::cast< clang::VarDecl >(expr->getDecl()->getUnderlyingDecl());
        }

        VarDeclOp getDefiningOpOfGlobalVar(const clang::VarDecl *decl) {
            return context().vars.lookup(decl).template getDefiningOp< VarDeclOp >();
        }

        Operation* VisitEnumDeclRefExpr(const clang::DeclRefExpr *expr) {
            auto decl = clang::cast< clang::EnumConstantDecl >(expr->getDecl()->getUnderlyingDecl());
            auto val = context().enumconsts.lookup(decl);
            auto rty = visit(expr->getType());
            return make< EnumRefOp >(meta_location(expr), rty, val.name());
        }

        Operation* VisitVarDeclRefExprImpl(const clang::DeclRefExpr *expr, Value var) {
            auto rty = getLValueReturnType(expr);
            return make< DeclRefOp >(meta_location(expr), rty, var);
        }

        Operation* VisitVarDeclRefExpr(const clang::DeclRefExpr *expr) {
            auto decl = getDeclForVarRef(expr);
            return VisitVarDeclRefExprImpl(expr, context().vars.lookup(decl));
        }

        Operation* VisitFileVarDeclRefExpr(const clang::DeclRefExpr *expr) {
            auto decl = getDeclForVarRef(expr);
            auto var  = getDefiningOpOfGlobalVar(decl);
            auto name = mlir::StringAttr::get(&mcontext(), var.name());

            auto rty = getLValueReturnType(expr);
            // reference to global variales first makes reference to global name, that makes
            // local SSA value that can be referenced the standard way as other variables
            return VisitVarDeclRefExprImpl(expr, make< GlobalRefOp >(meta_location(expr), rty, name));
        }

        Operation* VisitDeclRefExpr(const clang::DeclRefExpr *expr) {
            auto underlying = expr->getDecl()->getUnderlyingDecl();

            // TODO(Heno): deal with function declaration

            if (clang::isa< clang::EnumConstantDecl >(underlying)) {
                return VisitEnumDeclRefExpr(expr);
            }

            if (auto decl = clang::dyn_cast< clang::VarDecl >(underlying)) {
                if (decl->isFileVarDecl())
                    return VisitFileVarDeclRefExpr(expr);
                return VisitVarDeclRefExpr(expr);
            }

            VAST_UNREACHABLE("unknown underlying declaration to be referenced");
        }

        //
        // ControlFlow Statements
        //

        Operation* VisitReturnStmt(const clang::ReturnStmt *stmt) {
            auto loc = meta_location(stmt);
            if (auto ret = stmt->getRetValue())
                return make< ReturnOp >(loc, visit(ret)->getResults());
            return make< ReturnOp >(loc);
        }


        Operation* VisitBreakStmt(const clang::BreakStmt *stmt) {
            return make< BreakOp >(meta_location(stmt));
        }

        Operation* VisitContinueStmt(const clang::ContinueStmt *stmt) {
            return make< ContinueOp >(meta_location(stmt));
        }

        Operation* VisitCaseStmt(const clang::CaseStmt *stmt) {
            auto lhs_builder  = make_value_builder(stmt->getLHS());
            auto body_builder = make_region_builder(stmt->getSubStmt());
            return make< CaseOp >(meta_location(stmt), lhs_builder, body_builder);
        }

        Operation* VisitDefaultStmt(const clang::DefaultStmt *stmt) {
            auto body_builder = make_region_builder(stmt->getSubStmt());
            return make< DefaultOp >(meta_location(stmt), body_builder);
        }

        Operation* VisitSwitchStmt(const clang::SwitchStmt *stmt) {
            auto loc = meta_location(stmt);

            auto make_switch_op = [&] {
                auto cond_builder = make_value_builder(stmt->getCond());
                auto body_builder = make_region_builder(stmt->getBody());
                return make< SwitchOp >(loc, cond_builder, body_builder);
            };

            if (stmt->getInit()) {
                return this->template make_scoped< HighLevelScope >(loc, [&] {
                    visit(stmt->getInit());
                    make_switch_op();
                });
            }

            return make_switch_op();
        }

        Operation* VisitDoStmt(const clang::DoStmt *stmt) {
            auto cond_builder = make_cond_builder(stmt->getCond());
            auto body_builder = make_region_builder(stmt->getBody());
            return make< DoOp >(meta_location(stmt), body_builder, cond_builder);
        }

        Operation* VisitWhileStmt(const clang::WhileStmt *stmt) {
            auto cond_builder = make_cond_builder(stmt->getCond());
            auto body_builder = make_region_builder(stmt->getBody());
            return make< WhileOp >(meta_location(stmt), cond_builder, body_builder);
        }

        // Operation* VisitCXXCatchStmt(const clang::CXXCatchStmt *stmt)
        // Operation* VisitCXXForRangeStmt(const clang::CXXForRangeStmt *stmt)
        // Operation* VisitCXXTryStmt(const clang::CXXTryStmt *stmt)
        // Operation* VisitCXXTryStmt(const clang::CXXTryStmt *stmt)
        // Operation* VisitCapturedStmt(const clang::CapturedStmt *stmt)

        Operation* VisitForStmt(const clang::ForStmt *stmt) {
            auto loc = meta_location(stmt);

            auto make_loop_op = [&] {
                auto incr = make_region_builder(stmt->getInc());
                auto body = make_region_builder(stmt->getBody());
                if (auto cond = stmt->getCond())
                    return make< ForOp >(loc, make_cond_builder(cond), incr, body);
                return make< ForOp >(loc, make_yield_true(), incr, body);
            };

            if (stmt->getInit()) {
                return this->template make_scoped< HighLevelScope >(loc, [&] {
                    visit(stmt->getInit());
                    make_loop_op();
                });
            }

            return make_loop_op();
        }

        Operation* VisitGotoStmt(const clang::GotoStmt *stmt) {
            auto lab = visit(stmt->getLabel())->getResult(0);
            return make< GotoStmt >(meta_location(stmt), lab);
        }
        // Operation* VisitIndirectGotoStmt(const clang::IndirectGotoStmt *stmt)

        Operation* VisitLabelStmt(const clang::LabelStmt *stmt) {
            auto lab = visit(stmt->getDecl())->getResult(0);
            auto sub_builder = make_region_builder(stmt->getSubStmt());
            return make< LabelStmt >(meta_location(stmt), lab, sub_builder);
        }

        Operation* VisitIfStmt(const clang::IfStmt *stmt) {
            return this->template make_operation< IfOp >()
                .bind(meta_location(stmt))
                .bind(make_cond_builder(stmt->getCond()))
                .bind(make_region_builder(stmt->getThen()))
                .bind_if(stmt->getElse(), make_region_builder(stmt->getElse()))
                .freeze();
        }

        //
        // Expressions
        //

        Operation* VisitMemberExpr(const clang::MemberExpr *expr) {
            auto name = context().get_decl_name(expr->getMemberDecl());
            auto base = visit(expr->getBase())->getResult(0);
            auto type = visit_as_lvalue_type(expr->getType());
            return make< RecordMemberOp >(meta_location(expr), type, base, name);
        }

        // Operation* VisitAbstractConditionalOperator(const clang::AbstractConditionalOperator *op)
        // Operation* VisitAbstractConditionalOperator(const clang::BinaryConditionalOperator *op)
        // Operation* VisitConditionalOperator(const clang::ConditionalOperator *op)
        Operation* VisitAddrLabelExpr(const clang::AddrLabelExpr *expr) {
            auto lab = visit(expr->getLabel())->getResult(0);
            auto rty = visit_as_lvalue_type(expr->getType());
            return make< AddrLabelExpr >(meta_location(expr), rty, lab);
        }

        Operation* VisitConstantExpr(const clang::ConstantExpr *expr) {
            // TODO(Heno): crete hl.constantexpr
            return visit(expr->getSubExpr());
        }

        Operation* VisitArraySubscriptExpr(const clang::ArraySubscriptExpr *expr) {
            auto rty    = visit_as_lvalue_type(expr->getType());
            auto base   = visit(expr->getBase())->getResult(0);
            auto offset = visit(expr->getIdx())->getResult(0);
            return make< SubscriptOp >(meta_location(expr), rty, base, offset);
        }

        // Operation* VisitArrayTypeTraitExpr(const clang::ArrayTypeTraitExpr *expr)
        // Operation* VisitAsTypeExpr(const clang::AsTypeExpr *expr)
        // Operation* VisitAtomicExpr(const clang::AtomicExpr *expr)
        // Operation* VisitBlockExpr(const clang::BlockExpr *expr)

        // Operation* VisitCXXBindTemporaryExpr(const clang::CXXBindTemporaryExpr *expr)

        Operation* VisitCXXBoolLiteralExpr(const clang::CXXBoolLiteralExpr *lit) {
            return VisitScalarLiteral(lit, lit->getValue());
        }

        // Operation* VisitCXXConstructExpr(const clang::CXXConstructExpr *expr)
        // Operation* VisitCXXTemporaryObjectExpr(const clang::CXXTemporaryObjectExpr *expr)
        // Operation* VisitCXXDefaultArgExpr(const clang::CXXDefaultArgExpr *expr)
        // Operation* VisitCXXDefaultInitExpr(const clang::CXXDefaultInitExpr *expr)
        // Operation* VisitCXXDeleteExpr(const clang::CXXDeleteExpr *expr)
        // Operation* VisitCXXDependentScopeMemberExpr(const clang::CXXDependentScopeMemberExpr *expr)
        // Operation* VisitCXXNewExpr(const clang::CXXNewExpr *expr)
        // Operation* VisitCXXNoexceptExpr(const clang::CXXNoexceptExpr *expr)
        // Operation* VisitCXXNullPtrLiteralExpr(const clang::CXXNullPtrLiteralExpr *expr)
        // Operation* VisitCXXPseudoDestructorExpr(const clang::CXXPseudoDestructorExpr *expr)
        // Operation* VisitCXXScalarValueInitExpr(const clang::CXXScalarValueInitExpr *expr)
        // Operation* VisitCXXStdInitializerListExpr(const clang::CXXStdInitializerListExpr *expr)
        // Operation* VisitCXXThisExpr(const clang::CXXThisExpr *expr)
        // Operation* VisitCXXThrowExpr(const clang::CXXThrowExpr *expr)
        // Operation* VisitCXXTypeidExpr(const clang::CXXTypeidExpr *expr)
        // Operation* CXXFoldExpr(const clang::CXXFoldExpr *expr)
        // Operation* VisitCXXUnresolvedConstructExpr(const clang::CXXThrowExpr *expr)
        // Operation* VisitCXXUuidofExpr(const clang::CXXUuidofExpr *expr)

        mlir::FuncOp VisitDirectCallee(const clang::FunctionDecl *callee) {
            InsertionGuard guard(op_builder());

            if (auto fn = context().lookup_function(callee, false /* with error */)) {
                return fn;
            }

            set_insertion_point_to_start(&context().getBodyRegion());
            return mlir::cast< mlir::FuncOp >(visit(callee));
        }

        Operation* VisitIndirectCallee(const clang::Expr *callee) {
            return visit(callee);
        }

        using Arguments = llvm::SmallVector< Value, 2 >;

        Arguments VisitArguments(const clang::CallExpr *expr) {
            Arguments args;
            for (const auto &arg : expr->arguments()) {
                args.push_back(visit(arg)->getResult(0));
            }
            return args;
        }

        Operation* VisitDirectCall(const clang::CallExpr *expr) {
            auto callee = VisitDirectCallee(expr->getDirectCallee());
            auto args   = VisitArguments(expr);
            return make< CallOp >(meta_location(expr), callee, args);
        }

        Operation* VisitIndirectCall(const clang::CallExpr *expr) {
            auto callee = VisitIndirectCallee(expr->getCallee())->getResult(0);
            auto args   = VisitArguments(expr);
            return make< IndirectCallOp >(meta_location(expr), callee, args);
        }

        Operation* VisitCallExpr(const clang::CallExpr *expr) {
            if (expr->getDirectCallee()) {
                return VisitDirectCall(expr);
            }

            return VisitIndirectCall(expr);
        }

        // Operation* VisitCXXMemberCallExpr(const clang::CXXMemberCallExpr *expr)
        // Operation* VisitCXXOperatorCallExpr(const clang::CXXOperatorCallExpr *expr)

        // Operation* VisitOffsetOfExpr(const clang::OffsetOfExpr *expr)
        // Operation* VisitOpaqueValueExpr(const clang::OpaqueValueExpr *expr)
        // Operation* VisitOverloadExpr(const clang::OverloadExpr *expr)

        Operation* VisitParenExpr(const clang::ParenExpr *expr) {
            auto [region, type] = make_value_yield_region(expr->getSubExpr());
            return make< ExprOp >(meta_location(expr), type, std::move(region));
        }

        // Operation* VisitParenListExpr(const clang::ParenListExpr *expr)
        // Operation* VisitStmtExpr(const clang::StmtExpr *expr)

        template< typename Op >
        Operation* ExprTypeTrait(const clang::UnaryExprOrTypeTraitExpr *expr, auto rty, auto loc) {
            auto arg = make_value_builder(expr->getArgumentExpr());
            return make< Op >(loc, rty, arg);
        }

        template< typename Op >
        Operation* TypeTraitExpr(const clang::UnaryExprOrTypeTraitExpr *expr, auto rty, auto loc) {
            auto arg = visit(expr->getArgumentType());
            return make< Op >(loc, rty, arg);
        }

        template< typename TypeTraitOp, typename ExprTraitOp >
        Operation* VisitTraitExpr(const clang::UnaryExprOrTypeTraitExpr *expr) {
            auto loc = meta_location(expr);
            auto rty = visit(expr->getType());

            return expr->isArgumentType() ? TypeTraitExpr< TypeTraitOp >(expr, rty, loc)
                                          : ExprTypeTrait< ExprTraitOp >(expr, rty, loc);
        }

        Operation* VisitUnaryExprOrTypeTraitExpr(const clang::UnaryExprOrTypeTraitExpr *expr) {
            auto kind = expr->getKind();

            if (kind == clang::UETT_SizeOf) {
                return VisitTraitExpr< SizeOfTypeOp, SizeOfExprOp >(expr);
            }

            if (kind == clang::UETT_AlignOf) {
                return VisitTraitExpr< AlignOfTypeOp, AlignOfExprOp >(expr);
            }

            VAST_UNREACHABLE("unsupported UnaryExprOrTypeTraitExpr");
        }

        // Operation* VisitVAArgExpr(const clang::VAArgExpr *expr)

        Operation* VisitNullStmt(const clang::NullStmt *stmt) {
            return make< SkipStmt >(meta_location(stmt));
        }

        //
        // Literals
        //

        template< typename LiteralType, typename Value >
        Operation* VisitScalarLiteral(const LiteralType *lit, Value value) {
            auto type = visit(lit->getType());
            return constant(meta_location(lit), type, value).getDefiningOp();
        }

        Operation* VisitCharacterLiteral(const clang::CharacterLiteral *lit) {
            return VisitScalarLiteral(lit, lit->getValue());
        }

        Operation* VisitIntegerLiteral(const clang::IntegerLiteral *lit) {
            return VisitScalarLiteral(lit, lit->getValue());
        }

        Operation* VisitFloatingLiteral(const clang::FloatingLiteral *lit) {
            return VisitScalarLiteral(lit, lit->getValue());
        }

        Operation* VisitStringLiteral(const clang::StringLiteral *lit) {
            return VisitScalarLiteral(lit, lit->getString());
        }

        // Operation* VisitUserDefinedLiteral(const clang::UserDefinedLiteral *lit)
        // Operation* VisitCompoundLiteralExpr(const clang::CompoundLiteralExpr *lit)
        // Operation* VisitFixedPointLiteral(const clang::FixedPointLiteral *lit)

        Operation* VisitInitListExpr(const clang::InitListExpr *expr) {
            auto ty = visit(expr->getType());

            llvm::SmallVector< Value > elements;
            for (auto elem : expr->inits()) {
                elements.push_back(visit(elem)->getResult(0));
            }

            return make< InitListExpr >(meta_location(expr), ty, elements);
        }
    };

} // namespace vast::hl
