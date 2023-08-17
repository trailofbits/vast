// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <clang/AST/StmtVisitor.h>
VAST_UNRELAX_WARNINGS

#include "vast/CodeGen/CodeGenMeta.hpp"
#include "vast/CodeGen/CodeGenBuilder.hpp"
#include "vast/CodeGen/CodeGenVisitorBase.hpp"
#include "vast/CodeGen/CodeGenVisitorLens.hpp"

#include "vast/Dialect/HighLevel/HighLevelDialect.hpp"
#include "vast/Dialect/HighLevel/HighLevelOps.hpp"

#include "vast/Util/Common.hpp"

namespace vast::hl {

    CastKind cast_kind(const clang::CastExpr *expr);
    IdentKind ident_kind(const clang::PredefinedExpr *expr);

} // namespace vast::hl


namespace vast::cg {

    template< typename Derived >
    struct CodeGenStmtVisitor
        : clang::ConstStmtVisitor< CodeGenStmtVisitor< Derived >, Operation* >
        , CodeGenVisitorLens< CodeGenStmtVisitor< Derived >, Derived >
        , CodeGenBuilder< CodeGenStmtVisitor< Derived >, Derived >
    {
        using LensType = CodeGenVisitorLens< CodeGenStmtVisitor< Derived >, Derived >;

        using LensType::derived;
        using LensType::context;
        using LensType::mcontext;

        using LensType::meta_location;

        using LensType::visit;
        using LensType::visit_as_lvalue_type;

        using Builder = CodeGenBuilder< CodeGenStmtVisitor< Derived >, Derived >;

        using Builder::builder;

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
            auto type = visit(op->getType());
            return make< Op >(meta_location(op), type, lhs, rhs);
        }

        template< typename IOp, typename FOp >
        Operation* VisitIFBinOp(const clang::BinaryOperator *op) {
            auto ty = op->getType();
            if (ty->isIntegerType())
                return VisitBinOp< IOp >(op);
            // FIXME: eventually decouple arithmetic and pointer additions?
            if (ty->isPointerType())
                return VisitBinOp< IOp >(op);
            if (ty->isFloatingType())
                return VisitBinOp< FOp >(op);
            return nullptr;
        }

        template< typename UOp, typename SOp, typename FOp >
        Operation* VisitIFBinOp(const clang::BinaryOperator *op) {
            auto ty = op->getType();
            if (ty->isUnsignedIntegerType())
                return VisitBinOp< UOp >(op);
            if (ty->isIntegerType())
                return VisitBinOp< SOp >(op);
            if (ty->isFloatingType())
                return VisitBinOp< FOp >(op);
            return nullptr;
        }

        template< hl::Predicate pred >
        Operation* VisitCmp(const clang::BinaryOperator *op) {
            auto lhs = visit(op->getLHS())->getResult(0);
            auto rhs = visit(op->getRHS())->getResult(0);
            auto res = visit(op->getType());
            return make< hl::CmpOp >(meta_location(op), res, pred, lhs, rhs);
        }

        template< hl::FPredicate pred >
        Operation* VisitFCmp(const clang::BinaryOperator *op) {
            auto lhs = visit(op->getLHS())->getResult(0);
            auto rhs = visit(op->getRHS())->getResult(0);
            auto res = visit(op->getType());
            return make< hl::FCmpOp >(meta_location(op), res, pred, lhs, rhs);
        }

        template< hl::Predicate upred, hl::Predicate spred, hl::FPredicate fpred >
        Operation* VisitCmp(const clang::BinaryOperator *op) {
            auto ty = op->getLHS()->getType();
            if (ty->isUnsignedIntegerType())
                return VisitCmp< upred >(op);
            if (ty->isPointerType())
                return VisitCmp< upred >(op);
            if (ty->isIntegerType())
                return VisitCmp< spred >(op);
            if (ty->isFloatingType())
                return VisitFCmp< fpred >(op);
            return nullptr;
        }

        // Operation* VisitBinPtrMemD(clang::BinaryOperator *op);
        // Operation* VisitBinPtrMemI(clang::BinaryOperator *op);

        Operation* VisitBinMul(const clang::BinaryOperator *op) {
            return VisitIFBinOp< hl::MulIOp, hl::MulFOp >(op);
        }

        Operation* VisitBinDiv(const clang::BinaryOperator *op) {
            return VisitIFBinOp< hl::DivUOp, hl::DivSOp, hl::DivFOp >(op);
        }

        Operation* VisitBinRem(const clang::BinaryOperator *op) {
            return VisitIFBinOp< hl::RemUOp, hl::RemSOp, hl::RemFOp >(op);
        }

        Operation* VisitBinAdd(const clang::BinaryOperator *op) {
            return VisitIFBinOp< hl::AddIOp, hl::AddFOp >(op);
        }

        Operation* VisitBinSub(const clang::BinaryOperator *op) {
            return VisitIFBinOp< hl::SubIOp, hl::SubFOp >(op);
        }

        Operation* VisitBinShl(const clang::BinaryOperator *op) {
            return VisitBinOp< hl::BinShlOp >(op);
        }

        Operation* VisitBinShr(const clang::BinaryOperator *op) {
            return VisitBinOp< hl::BinShrOp >(op);
        }

        using ipred = hl::Predicate;
        using fpred = hl::FPredicate;

        Operation* VisitBinLT(const clang::BinaryOperator *op) {
            return VisitCmp< ipred::ult, ipred::slt, fpred::olt >(op);
        }

        Operation* VisitBinGT(const clang::BinaryOperator *op) {
            return VisitCmp< ipred::ugt, ipred::sgt, fpred::ogt >(op);
        }

        Operation* VisitBinLE(const clang::BinaryOperator *op) {
            return VisitCmp< ipred::ule, ipred::sle, fpred::ole >(op);
        }

        Operation* VisitBinGE(const clang::BinaryOperator *op) {
            return VisitCmp< ipred::uge, ipred::sge, fpred::oge >(op);
        }

        Operation* VisitBinEQ(const clang::BinaryOperator *op) {
            return VisitCmp< ipred::eq, ipred::eq, fpred::oeq >(op);
        }

        Operation* VisitBinNE(const clang::BinaryOperator *op) {
            return VisitCmp< ipred::ne, ipred::ne, fpred::une >(op);
        }

        Operation* VisitBinAnd(const clang::BinaryOperator *op) {
            return VisitBinOp< hl::BinAndOp >(op);
        }

        Operation* VisitBinXor(const clang::BinaryOperator *op) {
            return VisitBinOp< hl::BinXorOp >(op);
        }

        Operation* VisitBinOr(const clang::BinaryOperator *op) {
            return VisitBinOp< hl::BinOrOp >(op);
        }

        template< typename LOp >
        Operation* VisitBinLogical(const clang::BinaryOperator *op) {
            auto lhs_builder = make_value_builder(op->getLHS());
            auto rhs_builder = make_value_builder(op->getRHS());
            auto type = visit(op->getType());
            return make< LOp >(meta_location(op), type, lhs_builder, rhs_builder);
        }

        Operation* VisitBinLAnd(const clang::BinaryOperator *op) {
            return VisitBinLogical< hl::BinLAndOp >(op);
        }

        Operation* VisitBinLOr(const clang::BinaryOperator *op) {
            return VisitBinLogical< hl::BinLOrOp >(op);
        }

        template< typename Op >
        Operation* VisitAssignBinOp(const clang::BinaryOperator *op) {
            auto lhs = visit(op->getLHS())->getResult(0);
            auto rhs = visit(op->getRHS())->getResult(0);
            return make< Op >(meta_location(op), lhs, rhs);
        }

        template< typename UOp, typename SOp >
        Operation* VisitAssignIBinOp(const clang::BinaryOperator *op) {
            auto ty = op->getType();
            if (ty->isUnsignedIntegerType())
                return VisitAssignBinOp< UOp >(op);
            if (ty->isIntegerType())
                return VisitAssignBinOp< SOp >(op);
            return nullptr;
        }

        Operation* VisitBinAssign(const clang::BinaryOperator *op) {
            return VisitAssignBinOp< hl::AssignOp >(op);
        }

        //
        // Compound Assignment Operations
        //

        Operation* VisitBinMulAssign(const clang::CompoundAssignOperator *op) {
            return VisitAssignBinOp< hl::MulIAssignOp >(op);
        }

        Operation* VisitBinDivAssign(const clang::CompoundAssignOperator *op) {
            return VisitAssignIBinOp< hl::DivUAssignOp, hl::DivSAssignOp >(op);
        }

        Operation* VisitBinRemAssign(const clang::CompoundAssignOperator *op) {
            return VisitAssignIBinOp< hl::RemUAssignOp, hl::RemSAssignOp >(op);
        }

        Operation* VisitBinAddAssign(const clang::CompoundAssignOperator *op) {
            return VisitAssignBinOp< hl::AddIAssignOp >(op);
        }

        Operation* VisitBinSubAssign(const clang::CompoundAssignOperator *op) {
            return VisitAssignBinOp< hl::SubIAssignOp >(op);
        }

        Operation* VisitBinShlAssign(const clang::CompoundAssignOperator *op) {
            return VisitAssignBinOp< hl::BinShlAssignOp >(op);
        }

        Operation* VisitBinShrAssign(const clang::CompoundAssignOperator *op) {
            return VisitAssignBinOp< hl::BinShrAssignOp >(op);
        }

        Operation* VisitBinAndAssign(const clang::CompoundAssignOperator *op) {
            return VisitAssignBinOp< hl::BinAndAssignOp >(op);
        }

        Operation* VisitBinOrAssign(const clang::CompoundAssignOperator *op) {
            return VisitAssignBinOp< hl::BinOrAssignOp >(op);
        }

        Operation* VisitBinXorAssign(const clang::CompoundAssignOperator *op) {
            return VisitAssignBinOp< hl::BinXorAssignOp >(op);
        }

        Operation* VisitBinComma(const clang::BinaryOperator *op) {
            auto lhs_op = visit(op->getLHS());
            auto rhs_op = visit(op->getRHS());
            auto ty  = visit(op->getType());
            return make< hl::BinComma >(meta_location(op), ty, lhs_op, rhs_op);
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
            if (auto ltype = type.template dyn_cast< hl::LValueType >()) {
                type = ltype.getElementType();
            }
            return make< Op >(meta_location(op), type, arg);
        }

        Operation* VisitUnaryPostInc(const clang::UnaryOperator *op) {
            return VisitUnderlyingTypePreservingUnary< hl::PostIncOp >(op);
        }

        Operation* VisitUnaryPostDec(const clang::UnaryOperator *op) {
            return VisitUnderlyingTypePreservingUnary< hl::PostDecOp >(op);
        }

        Operation* VisitUnaryPreInc(const clang::UnaryOperator *op) {
            return VisitUnderlyingTypePreservingUnary< hl::PreIncOp >(op);
        }

        Operation* VisitUnaryPreDec(const clang::UnaryOperator *op) {
            return VisitUnderlyingTypePreservingUnary< hl::PreDecOp >(op);
        }

        Operation* VisitUnaryAddrOf(const clang::UnaryOperator *op) {
            return VisitUnary< hl::AddressOf >(op, visit(op->getType()));
        }

        Operation* VisitUnaryDeref(const clang::UnaryOperator *op) {
            return VisitUnary< hl::Deref >(op, visit_as_lvalue_type(op->getType()));
        }

        Operation* VisitUnaryPlus(const clang::UnaryOperator *op) {
            return VisitUnderlyingTypePreservingUnary< hl::PlusOp >(op);
        }

        Operation* VisitUnaryMinus(const clang::UnaryOperator *op) {
            return VisitUnderlyingTypePreservingUnary< hl::MinusOp >(op);
        }

        Operation* VisitUnaryNot(const clang::UnaryOperator *op) {
            return VisitUnderlyingTypePreservingUnary< hl::NotOp >(op);
        }

        Operation* VisitUnaryLNot(const clang::UnaryOperator *op) {
            return VisitUnary< hl::LNotOp >(op, visit(op->getType()));
        }

        Operation* VisitUnaryExtension(const clang::UnaryOperator *op) {
            auto visited = visit(op->getSubExpr());
            // TODO(void-call): Unnecessary condition
            if (visited->getNumResults() > 0) {
                auto arg = visited->getResult(0);
                return make< hl::ExtensionOp >(meta_location(op), arg.getType(), arg);
            }
            return make< hl::ExtensionOp >(meta_location(op), mlir::Type(), mlir::Value());
        }

        // Operation* VisitUnaryReal(const clang::UnaryOperator *op)
        // Operation* VisitUnaryImag(const clang::UnaryOperator *op)
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
            auto array_to_ptr_cast  = [&] { return visit(expr->getType()); };
            auto keep_category_cast = [&] {
                if (mlir::isa< hl::LValueType >(from))
                    return lvalue_cast();
                return non_lvalue_cast();
            };

            auto unsupported_cast   = [&] { return visit(expr->getType()); };

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
                case clang::CastKind::CK_ArrayToPointerDecay:    return array_to_ptr_cast();
                case clang::CastKind::CK_FunctionToPointerDecay:
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
                default: return unsupported_cast();
            }
        }


        template< typename Cast >
        Operation* VisitCast(const clang::CastExpr *expr) {
            auto arg = visit(expr->getSubExpr());
            auto rty = VisitCastReturnType(expr, arg->getResultTypes().front());
            return make< Cast >(meta_location(expr), rty, arg->getResult(0), hl::cast_kind(expr));
        }

        Operation* VisitImplicitCastExpr(const clang::ImplicitCastExpr *expr) {
             return VisitCast< hl::ImplicitCastOp >(expr);
        }

        Operation* VisitCStyleCastExpr(const clang::CStyleCastExpr *expr) {
            return VisitCast< hl::CStyleCastOp >(expr);
        }

        Operation* VisitBuiltinBitCastExpr(const clang::BuiltinBitCastExpr *expr) {
            return VisitCast< hl::BuiltinBitCastOp >(expr);
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

        hl::VarDeclOp getDefiningOpOfGlobalVar(const clang::VarDecl *decl) {
            return context().vars.lookup(decl).template getDefiningOp< hl::VarDeclOp >();
        }

        Operation* VisitEnumDeclRefExpr(const clang::DeclRefExpr *expr) {
            auto decl = clang::cast< clang::EnumConstantDecl >(expr->getDecl()->getUnderlyingDecl());
            auto val = context().enumconsts.lookup(decl);
            auto rty = visit(expr->getType());
            return make< hl::EnumRefOp >(meta_location(expr), rty, val.getName());
        }

        Operation* VisitVarDeclRefExprImpl(const clang::DeclRefExpr *expr, Value var) {
            auto rty = getLValueReturnType(expr);
            return make< hl::DeclRefOp >(meta_location(expr), rty, var);
        }

        Operation* VisitVarDeclRefExpr(const clang::DeclRefExpr *expr) {
            auto decl = getDeclForVarRef(expr);
            return VisitVarDeclRefExprImpl(expr, context().vars.lookup(decl));
        }

        Operation* VisitFileVarDeclRefExpr(const clang::DeclRefExpr *expr) {
            auto decl = getDeclForVarRef(expr);
            auto var  = getDefiningOpOfGlobalVar(decl);
            auto name = mlir::StringAttr::get(&mcontext(), var.getName());

            auto rty = getLValueReturnType(expr);
            return make< hl::GlobalRefOp >(meta_location(expr), rty, name);
        }

        Operation* VisitFunctionDeclRefExpr(const clang::DeclRefExpr *expr) {
            auto decl = clang::cast< clang::FunctionDecl >( expr->getDecl()->getUnderlyingDecl() );
            auto mangled = context().get_mangled_name(decl);
            auto fn      = context().lookup_function(mangled, false);
            if (!fn) {
                InsertionGuard guard(builder());
                set_insertion_point_to_start(&context().getBodyRegion());
                fn = mlir::cast< hl::FuncOp >(visit(decl));
            }
            auto rty = getLValueReturnType(expr);

            return make< hl::FuncRefOp >(meta_location(expr), rty, mlir::SymbolRefAttr::get(fn));
        }

        Operation* VisitDeclRefExpr(const clang::DeclRefExpr *expr) {
            auto underlying = expr->getDecl()->getUnderlyingDecl();

            if (clang::isa< clang::EnumConstantDecl >(underlying)) {
                return VisitEnumDeclRefExpr(expr);
            }

            if (auto decl = clang::dyn_cast< clang::VarDecl >(underlying)) {
                if (decl->isFileVarDecl())
                    return VisitFileVarDeclRefExpr(expr);
                return VisitVarDeclRefExpr(expr);
            }

            if (clang::isa< clang::FunctionDecl >(underlying)) {
                return VisitFunctionDeclRefExpr(expr);
            }

            VAST_UNREACHABLE("unknown underlying declaration to be referenced");
        }

        Operation *VisitPredefinedExpr(const clang::PredefinedExpr *expr)
        {
            auto name = expr->getFunctionName();
            VAST_CHECK(name, "clang::PredefinedExpr without name has missing support.");

            auto name_as_op = this->VisitStringLiteral(name)->getResult(0);
            auto kind = hl::ident_kind( expr );

            return make< hl::PredefinedExpr >(meta_location(expr),
                                              name_as_op.getType(), name_as_op, kind);
        }

        //
        // ControlFlow Statements
        //

        Operation* VisitReturnStmt(const clang::ReturnStmt *stmt) {
            auto loc = meta_location(stmt);
            if (auto ret = stmt->getRetValue())
                return make< hl::ReturnOp >(loc, visit(ret)->getResults());
            return make< hl::ReturnOp >(loc);
        }


        Operation* VisitBreakStmt(const clang::BreakStmt *stmt) {
            return make< hl::BreakOp >(meta_location(stmt));
        }

        Operation* VisitContinueStmt(const clang::ContinueStmt *stmt) {
            return make< hl::ContinueOp >(meta_location(stmt));
        }

        Operation* VisitCaseStmt(const clang::CaseStmt *stmt) {
            auto lhs_builder  = make_value_builder(stmt->getLHS());
            auto body_builder = make_region_builder(stmt->getSubStmt());
            return make< hl::CaseOp >(meta_location(stmt), lhs_builder, body_builder);
        }

        Operation* VisitDefaultStmt(const clang::DefaultStmt *stmt) {
            auto body_builder = make_region_builder(stmt->getSubStmt());
            return make< hl::DefaultOp >(meta_location(stmt), body_builder);
        }

        Operation* VisitSwitchStmt(const clang::SwitchStmt *stmt) {
            auto loc = meta_location(stmt);

            auto make_switch_op = [&] {
                auto cond_builder = make_value_builder(stmt->getCond());
                auto body_builder = make_region_builder(stmt->getBody());
                return make< hl::SwitchOp >(loc, cond_builder, body_builder);
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
            return make< hl::DoOp >(meta_location(stmt), body_builder, cond_builder);
        }

        Operation* VisitWhileStmt(const clang::WhileStmt *stmt) {
            auto cond_builder = make_cond_builder(stmt->getCond());
            auto body_builder = make_region_builder(stmt->getBody());
            return make< hl::WhileOp >(meta_location(stmt), cond_builder, body_builder);
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
                    return make< hl::ForOp >(loc, make_cond_builder(cond), incr, body);
                return make< hl::ForOp >(loc, make_yield_true(), incr, body);
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
            return make< hl::GotoStmt >(meta_location(stmt), lab);
        }
        // Operation* VisitIndirectGotoStmt(const clang::IndirectGotoStmt *stmt)

        Operation* VisitLabelStmt(const clang::LabelStmt *stmt) {
            auto lab = visit(stmt->getDecl())->getResult(0);
            auto sub_builder = make_region_builder(stmt->getSubStmt());
            return make< hl::LabelStmt >(meta_location(stmt), lab, sub_builder);
        }

        Operation* VisitIfStmt(const clang::IfStmt *stmt) {
            return this->template make_operation< hl::IfOp >()
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
            return make< hl::RecordMemberOp >(meta_location(expr), type, base, name);
        }

        // Operation* VisitAbstractConditionalOperator(const clang::AbstractConditionalOperator *op)
        // Operation* VisitAbstractConditionalOperator(const clang::BinaryConditionalOperator *op)

        Operation* VisitConditionalOperator(const clang::ConditionalOperator *op) {
            auto type = visit(op->getType());
            auto cond = make_cond_builder(op->getCond());
            auto true_expr = make_value_builder(op->getTrueExpr());
            auto false_expr = make_value_builder(op->getFalseExpr());
            return make< hl::CondOp >(meta_location(op), type, cond, true_expr, false_expr);
        }

        Operation* VisitAddrLabelExpr(const clang::AddrLabelExpr *expr) {
            auto lab = visit(expr->getLabel())->getResult(0);
            auto rty = visit_as_lvalue_type(expr->getType());
            return make< hl::AddrLabelExpr >(meta_location(expr), rty, lab);
        }

        Operation* VisitConstantExpr(const clang::ConstantExpr *expr) {
            // TODO(Heno): crete hl.constantexpr
            return visit(expr->getSubExpr());
        }

        Operation* VisitArraySubscriptExpr(const clang::ArraySubscriptExpr *expr) {
            auto rty    = visit_as_lvalue_type(expr->getType());
            auto base   = visit(expr->getBase())->getResult(0);
            auto offset = visit(expr->getIdx())->getResult(0);
            return make< hl::SubscriptOp >(meta_location(expr), rty, base, offset);
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

        hl::FuncOp VisitDirectCallee(const clang::FunctionDecl *callee) {
            InsertionGuard guard(builder());

            auto mangled = context().get_mangled_name(callee);
            if (auto fn = context().lookup_function(mangled, false /* with error */)) {
                return fn;
            }

            set_insertion_point_to_start(&context().getBodyRegion());
            return mlir::cast< hl::FuncOp >(visit(callee));
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
            return make< hl::CallOp >(meta_location(expr), callee, args);
        }

        Operation* VisitIndirectCall(const clang::CallExpr *expr) {
            auto callee = VisitIndirectCallee(expr->getCallee())->getResult(0);
            auto args   = VisitArguments(expr);
            auto type   = hl::getFunctionType(callee.getType(), context().mod.get()).getResults();
            return make< hl::IndirectCallOp >(meta_location(expr), type, callee, args);
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
            auto [reg, rty] = make_value_yield_region(expr->getSubExpr());
            return make< hl::ExprOp >(meta_location(expr), rty, std::move(reg));
        }

        // Operation* VisitParenListExpr(const clang::ParenListExpr *expr)
        Operation* VisitStmtExpr(const clang::StmtExpr *expr) {
            auto loc = meta_location(expr);
            auto sub = llvm::cast< clang::CompoundStmt >(expr->getSubStmt());
            // TODO(void-call): Fix this funciton call
            auto [reg, rty] = Builder::make_maybe_value_yield_region(sub);
            return make< hl::StmtExprOp >(loc, rty, std::move(reg));
        }

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
                return VisitTraitExpr< hl::SizeOfTypeOp, hl::SizeOfExprOp >(expr);
            }

            if (kind == clang::UETT_AlignOf) {
                return VisitTraitExpr< hl::AlignOfTypeOp, hl::AlignOfExprOp >(expr);
            }

            VAST_UNREACHABLE("unsupported UnaryExprOrTypeTraitExpr");
        }

        // Operation* VisitVAArgExpr(const clang::VAArgExpr *expr)

        Operation* VisitNullStmt(const clang::NullStmt *stmt) {
            return make< hl::SkipStmt >(meta_location(stmt));
        }

        Operation* VisitCXXThisExpr(const clang::CXXThisExpr *expr) {
            auto rty = visit(expr->getType());
            return make< hl::ThisOp >(meta_location(expr), rty);
        }

        //
        // Literals
        //

        template< typename LiteralType, typename Value >
        Operation* VisitScalarLiteral(const LiteralType *lit, Value value) {
            if constexpr (std::is_same_v< Value, bool >) {
                return constant(meta_location(lit), value).getDefiningOp();
            } else {
                // in C string literals are arrays and therefore lvalues
                auto type = lit->isLValue() ? visit_as_lvalue_type(lit->getType())
                                            : visit(lit->getType());
                return constant(meta_location(lit), type, value).getDefiningOp();
            }
        }

        Operation* VisitCharacterLiteral(const clang::CharacterLiteral *lit) {
            return VisitScalarLiteral(lit, apsint(lit->getValue()));
        }

        Operation* VisitIntegerLiteral(const clang::IntegerLiteral *lit) {
            return VisitScalarLiteral(lit, llvm::APSInt(lit->getValue(), false));
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

            return make< hl::InitListExpr >(meta_location(expr), ty, elements);
        }
    };

} // namespace vast::cg
