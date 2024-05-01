// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <clang/AST/StmtVisitor.h>
#include <clang/AST/OperationKinds.h>
VAST_UNRELAX_WARNINGS

#include "vast/CodeGen/CodeGenVisitorBase.hpp"

namespace vast::cg {

    hl::CastKind cast_kind(const clang::CastExpr *expr);

    struct default_stmt_visitor : stmt_visitor_base< default_stmt_visitor >
    {
        using base = stmt_visitor_base< default_stmt_visitor >;
        using base::base;

        operation visit(const clang_stmt *stmt) {return Visit(stmt); }

        using base::Visit;

        operation VisitCompoundStmt(const clang::CompoundStmt *stmt);

        //
        // Binary Operations
        //
        template< typename Op >
        operation visit_bin_op(const clang::BinaryOperator *op);

        template< typename UOp, typename SOp >
        operation visit_ibin_op(const clang::BinaryOperator *op);

        template< typename UOp, typename SOp, typename FOp >
        operation visit_ifbin_op(const clang::BinaryOperator *op);

        template< hl::Predicate pred >
        operation visit_cmp_op(const clang::BinaryOperator *op);

        template< hl::FPredicate pred >
        operation visit_fcmp_op(const clang::BinaryOperator *op);

        template< hl::Predicate upred, hl::Predicate spred, hl::FPredicate fpred >
        operation visit_cmp_op(const clang::BinaryOperator *op);

        template< typename LOp >
        operation visit_logical_op(const clang::BinaryOperator *op);

        operation VisitBinPtrMemD(const clang::BinaryOperator *op);
        operation VisitBinPtrMemI(const clang::BinaryOperator *op);
        operation VisitBinMul(const clang::BinaryOperator *op);
        operation VisitBinDiv(const clang::BinaryOperator *op);
        operation VisitBinRem(const clang::BinaryOperator *op);
        operation VisitBinAdd(const clang::BinaryOperator *op);
        operation VisitBinSub(const clang::BinaryOperator *op);
        operation VisitBinShl(const clang::BinaryOperator *op);
        operation VisitBinShr(const clang::BinaryOperator *op);
        operation VisitBinLT(const clang::BinaryOperator *op);
        operation VisitBinGT(const clang::BinaryOperator *op);
        operation VisitBinLE(const clang::BinaryOperator *op);
        operation VisitBinGE(const clang::BinaryOperator *op);
        operation VisitBinEQ(const clang::BinaryOperator *op);
        operation VisitBinNE(const clang::BinaryOperator *op);
        operation VisitBinAnd(const clang::BinaryOperator *op);
        operation VisitBinXor(const clang::BinaryOperator *op);
        operation VisitBinOr(const clang::BinaryOperator *op);

        operation VisitBinLAnd(const clang::BinaryOperator *op);
        operation VisitBinLOr(const clang::BinaryOperator *op);

        //
        // Compound Assignment Operations
        //
        template< typename Op >
        operation visit_assign_bin_op(const clang::BinaryOperator *op);

        template< typename UOp, typename SOp >
        operation visit_assign_ibin_op(const clang::CompoundAssignOperator *op);

        template< typename UOp, typename SOp, typename FOp >
        operation visit_assign_ifbin_op(const clang::CompoundAssignOperator *op);

        operation VisitBinAssign(const clang::BinaryOperator *op);

        operation VisitBinMulAssign(const clang::CompoundAssignOperator *op);
        operation VisitBinDivAssign(const clang::CompoundAssignOperator *op);
        operation VisitBinRemAssign(const clang::CompoundAssignOperator *op);
        operation VisitBinAddAssign(const clang::CompoundAssignOperator *op);
        operation VisitBinSubAssign(const clang::CompoundAssignOperator *op);
        operation VisitBinShlAssign(const clang::CompoundAssignOperator *op);
        operation VisitBinShrAssign(const clang::CompoundAssignOperator *op);
        operation VisitBinAndAssign(const clang::CompoundAssignOperator *op);
        operation VisitBinXorAssign(const clang::CompoundAssignOperator *op);
        operation VisitBinOrAssign(const clang::CompoundAssignOperator *op);
        operation VisitBinComma(const clang::BinaryOperator *op);

        //
        // Unary Operations
        //
        template< typename Op >
        operation visit_unary_op(const clang::UnaryOperator *op, mlir_type rty);

        template< typename Op >
        operation visit_underlying_type_preserving_unary_op(const clang::UnaryOperator *op);

        operation VisitUnaryPostInc(const clang::UnaryOperator *op);
        operation VisitUnaryPostDec(const clang::UnaryOperator *op);
        operation VisitUnaryPreInc(const clang::UnaryOperator *op);
        operation VisitUnaryPreDec(const clang::UnaryOperator *op);
        operation VisitUnaryAddrOf(const clang::UnaryOperator *op);
        operation VisitUnaryDeref(const clang::UnaryOperator *op);
        operation VisitUnaryPlus(const clang::UnaryOperator *op);
        operation VisitUnaryMinus(const clang::UnaryOperator *op);
        operation VisitUnaryNot(const clang::UnaryOperator *op);
        operation VisitUnaryLNot(const clang::UnaryOperator *op);
        operation VisitUnaryExtension(const clang::UnaryOperator *op);
        operation VisitUnaryReal(const clang::UnaryOperator *op);
        operation VisitUnaryImag(const clang::UnaryOperator *op);
        operation VisitUnaryCoawait(const clang::UnaryOperator *op);

        //
        // Cast Operations
        //
        template< typename Op >
        operation visit_cast_op(const clang::CastExpr *cast);

        mlir_type cast_result_type(const clang::CastExpr *cast, mlir_type from);

        operation VisitImplicitCastExpr(const clang::ImplicitCastExpr *cast);
        operation VisitCStyleCastExpr(const clang::CStyleCastExpr *cast);
        operation VisitBuiltinBitCastExpr(const clang::BuiltinBitCastExpr *cast);

        operation VisitCXXFunctionalCastExpr(const clang::CXXFunctionalCastExpr *cast);
        operation VisitCXXConstCastExpr(const clang::CXXConstCastExpr *cast);
        operation VisitCXXDynamicCastExpr(const clang::CXXDynamicCastExpr *cast);
        operation VisitCXXReinterpretCastExpr(const clang::CXXReinterpretCastExpr *cast);
        operation VisitCXXStaticCastExpr(const clang::CXXStaticCastExpr *cast);

        //
        // Other Statements
        //
        operation VisitDeclRefExpr(const clang::DeclRefExpr *expr);

        operation visit_enum_decl_ref(const clang::DeclRefExpr *expr);
        operation visit_file_var_decl_ref(const clang::DeclRefExpr *expr);
        operation visit_var_decl_ref(const clang::DeclRefExpr *expr);
        operation visit_function_decl_ref(const clang::DeclRefExpr *expr);

        operation VisitPredefinedExpr(const clang::PredefinedExpr *expr);

        //
        // ControlFlow Statements
        //
        operation VisitReturnStmt(const clang::ReturnStmt *stmt);
        operation VisitBreakStmt(const clang::BreakStmt *stmt);
        operation VisitContinueStmt(const clang::ContinueStmt *stmt);
        operation VisitCaseStmt(const clang::CaseStmt *stmt);
        operation VisitDefaultStmt(const clang::DefaultStmt *stmt);
        operation VisitSwitchStmt(const clang::SwitchStmt *stmt);
        operation VisitDoStmt(const clang::DoStmt *stmt);
        // operation VisitCXXCatchStmt(const clang::CXXCatchStmt *stmt)
        // operation VisitCXXForRangeStmt(const clang::CXXForRangeStmt *stmt)
        // operation VisitCXXTryStmt(const clang::CXXTryStmt *stmt)
        // operation VisitCXXTryStmt(const clang::CXXTryStmt *stmt)
        // operation VisitCapturedStmt(const clang::CapturedStmt *stmt)
        operation VisitWhileStmt(const clang::WhileStmt *stmt);
        operation VisitForStmt(const clang::ForStmt *stmt);
        operation VisitGotoStmt(const clang::GotoStmt *stmt);
        operation VisitLabelStmt(const clang::LabelStmt *stmt);
        operation VisitIfStmt(const clang::IfStmt *stmt);

        //
        // Expressions
        //
        operation VisitDeclStmt(const clang::DeclStmt *stmt);

        operation VisitMemberExpr(const clang::MemberExpr *expr);
        operation VisitConditionalOperator(const clang::ConditionalOperator *op);
        operation VisitAddrLabelExpr(const clang::AddrLabelExpr *expr);
        operation VisitConstantExpr(const clang::ConstantExpr *expr);
        operation VisitArraySubscriptExpr(const clang::ArraySubscriptExpr *expr);
        // operation VisitArrayTypeTraitExpr(const clang::ArrayTypeTraitExpr *expr)
        // operation VisitAsTypeExpr(const clang::AsTypeExpr *expr)
        // operation VisitAtomicExpr(const clang::AtomicExpr *expr)
        // operation VisitBlockExpr(const clang::BlockExpr *expr)

        // operation VisitCXXBindTemporaryExpr(const clang::CXXBindTemporaryExpr *expr);
        operation VisitCXXBoolLiteralExpr(const clang::CXXBoolLiteralExpr *expr);

        // operation VisitCXXConstructExpr(const clang::CXXConstructExpr *expr)
        // operation VisitCXXTemporaryObjectExpr(const clang::CXXTemporaryObjectExpr *expr)
        // operation VisitCXXDefaultArgExpr(const clang::CXXDefaultArgExpr *expr)
        // operation VisitCXXDefaultInitExpr(const clang::CXXDefaultInitExpr *expr)
        // operation VisitCXXDeleteExpr(const clang::CXXDeleteExpr *expr)
        // operation VisitCXXDependentScopeMemberExpr(const clang::CXXDependentScopeMemberExpr *expr)
        // operation VisitCXXNewExpr(const clang::CXXNewExpr *expr)
        // operation VisitCXXNoexceptExpr(const clang::CXXNoexceptExpr *expr)
        // operation VisitCXXNullPtrLiteralExpr(const clang::CXXNullPtrLiteralExpr *expr)
        // operation VisitCXXPseudoDestructorExpr(const clang::CXXPseudoDestructorExpr *expr)
        // operation VisitCXXScalarValueInitExpr(const clang::CXXScalarValueInitExpr *expr)
        // operation VisitCXXStdInitializerListExpr(const clang::CXXStdInitializerListExpr *expr)
        // operation VisitCXXThisExpr(const clang::CXXThisExpr *expr)
        // operation VisitCXXThrowExpr(const clang::CXXThrowExpr *expr)
        // operation VisitCXXTypeidExpr(const clang::CXXTypeidExpr *expr)
        // operation CXXFoldExpr(const clang::CXXFoldExpr *expr)
        // operation VisitCXXUnresolvedConstructExpr(const clang::CXXThrowExpr *expr)
        // operation VisitCXXUuidofExpr(const clang::CXXUuidofExpr *expr)

        operation VisitCallExpr(const clang::CallExpr *expr);

        // operation VisitCXXMemberCallExpr(const clang::CXXMemberCallExpr *expr)
        // operation VisitCXXOperatorCallExpr(const clang::CXXOperatorCallExpr *expr)

        // operation VisitOffsetOfExpr(const clang::OffsetOfExpr *expr)
        // operation VisitOpaqueValueExpr(const clang::OpaqueValueExpr *expr)
        // operation VisitOverloadExpr(const clang::OverloadExpr *expr)

        operation VisitParenExpr(const clang::ParenExpr *expr);
        // operation VisitParenListExpr(const clang::ParenListExpr *expr)
        operation VisitStmtExpr(const clang::StmtExpr *expr);

        operation VisitUnaryExprOrTypeTraitExpr(const clang::UnaryExprOrTypeTraitExpr *expr);
        operation VisitVAArgExpr(const clang::VAArgExpr *expr);
        operation VisitNullStmt(const clang::NullStmt *stmt);
        operation VisitCXXThisExpr(const clang::CXXThisExpr *expr);

        //
        // Literals
        //
        operation VisistCharacterLiteral(const clang::CharacterLiteral *lit);
        operation VisitIntegerLiteral(const clang::IntegerLiteral *lit);
        operation VisitFloatingLiteral(const clang::FloatingLiteral *lit);
        operation VisitStringLiteral(const clang::StringLiteral *lit);
        operation VisitUserDefinedLiteral(const clang::UserDefinedLiteral *lit);
        operation VisitCompoundLiteralExpr(const clang::CompoundLiteralExpr *lit);
        operation VisitFixedPointLiteral(const clang::FixedPointLiteral *lit);

        template< typename yield_type >
        auto make_stmt_builder(const clang_stmt *stmt) {
            return [this, stmt] (auto &state, auto loc) {
                self.visit(stmt);
                auto &op = state.getBlock()->back();
                VAST_ASSERT(op.getNumResults() == 1);
                bld.create< yield_type >(loc, op.getResult(0));
            };
        }

        auto make_value_builder(const clang_stmt *stmt) {
            return make_stmt_builder< hl::ValueYieldOp >(stmt);
        }

        auto make_cond_builder(const clang_stmt *stmt) {
            return make_stmt_builder< hl::CondYieldOp >(stmt);
        }

        auto make_true_yielder() {
            return [this] (auto &, auto loc) {
               bld.create< hl::CondYieldOp >(loc, bld.true_value(loc));
            };
        }

        auto make_false_yielder() {
            return [this] (auto &, auto loc) {
                bld.create< hl::CondYieldOp >(loc, bld.false_value(loc));
            };
        }

        auto make_optional_region_builder(const clang_stmt *stmt) {
            return [this, stmt] (auto &bld, auto) {
                if (stmt) self.visit(stmt);
            };
        }
    };

    template< typename Op >
    operation default_stmt_visitor::visit_bin_op(const clang::BinaryOperator *op) {
        return bld.compose< Op >()
            .bind(self.location(op))
            .bind(self.visit(op->getType()))
            .bind_transform(self.visit(op->getLHS()), first_result)
            .bind_transform(self.visit(op->getRHS()), first_result)
            .freeze();
    }

    template< typename UOp, typename SOp >
    operation default_stmt_visitor::visit_ibin_op(const clang::BinaryOperator *op) {
        auto ty = op->getType();

        if (ty->isUnsignedIntegerType()) {
            return visit_bin_op< UOp >(op);
        } else if (ty->isIntegerType()) {
            return visit_bin_op< SOp >(op);
        } else {
            return {};
        }
    }

    template< typename UOp, typename SOp, typename FOp >
    operation default_stmt_visitor::visit_ifbin_op(const clang::BinaryOperator *op) {
        auto ty = op->getType();

        if (ty->isUnsignedIntegerType()) {
            return visit_bin_op< UOp >(op);
        } else if (ty->isIntegerType()) {
            return visit_bin_op< SOp >(op);
        } else if (ty->isFloatingType()) {
            return visit_bin_op< FOp >(op);
        } else {
            return {};
        }
    }

    template< hl::Predicate pred >
    operation default_stmt_visitor::visit_cmp_op(const clang::BinaryOperator *op) {
        return bld.compose< hl::CmpOp >()
            .bind(self.location(op))
            .bind(self.visit(op->getType()))
            .bind(pred)
            .bind_transform(self.visit(op->getLHS()), first_result)
            .bind_transform(self.visit(op->getRHS()), first_result)
            .freeze();
    }

    template< hl::FPredicate pred >
    operation default_stmt_visitor::visit_fcmp_op(const clang::BinaryOperator *op) {
        return bld.compose< hl::FCmpOp >()
            .bind(self.location(op))
            .bind(self.visit(op->getType()))
            .bind(pred)
            .bind_transform(self.visit(op->getLHS()), first_result)
            .bind_transform(self.visit(op->getRHS()), first_result)
            .freeze();
    }

    template< hl::Predicate upred, hl::Predicate spred, hl::FPredicate fpred >
    operation default_stmt_visitor::visit_cmp_op(const clang::BinaryOperator *op) {
        auto ty = op->getLHS()->getType();

        if (ty->isUnsignedIntegerType() || ty->isPointerType()) {
            return visit_cmp_op< upred >(op);
        } else if (ty->isIntegerType()) {
            return visit_cmp_op< spred >(op);
        } else if (ty->isFloatingType()) {
            return visit_fcmp_op< fpred >(op);
        } else {
            return {};
        }
    }

    template< typename LOp >
    operation default_stmt_visitor::visit_logical_op(const clang::BinaryOperator *op) {
        return bld.compose< LOp >()
            .bind(self.location(op))
            .bind(self.visit(op->getType()))
            .bind_region(make_value_builder(op->getLHS()))
            .bind_region(make_value_builder(op->getRHS()))
            .freeze();
    }

    template< typename Op >
    operation default_stmt_visitor::visit_assign_bin_op(const clang::BinaryOperator *op) {
        return bld.compose< Op >()
            .bind(self.location(op))
            .bind_transform(self.visit(op->getLHS()), first_result)
            .bind_transform(self.visit(op->getRHS()), first_result)
            .freeze();
    }

    template< typename UOp, typename SOp >
    operation default_stmt_visitor::visit_assign_ibin_op(const clang::CompoundAssignOperator *op) {
        auto ty = op->getType();

        if (ty->isUnsignedIntegerType()) {
            return visit_assign_bin_op< UOp >(op);
        } else if (ty->isIntegerType()) {
            return visit_assign_bin_op< SOp >(op);
        } else {
            return {};
        }

    }

    template< typename UOp, typename SOp, typename FOp >
    operation default_stmt_visitor::visit_assign_ifbin_op(const clang::CompoundAssignOperator *op) {
        auto ty = op->getType();

        if (ty->isUnsignedIntegerType()) {
            return visit_assign_bin_op< UOp >(op);
        } else if (ty->isIntegerType()) {
            return visit_assign_bin_op< SOp >(op);
        } else if (ty->isFloatingType()) {
            return visit_assign_bin_op< FOp >(op);
        } else {
            return {};
        }
    }

    template< typename Op >
    operation default_stmt_visitor::visit_unary_op(const clang::UnaryOperator *op, mlir_type rty) {
        return bld.compose< Op >()
            .bind(self.location(op))
            .bind(rty)
            .bind_transform(self.visit(op->getSubExpr()), first_result)
            .freeze();
    }

    static inline mlir_type strip_lvalue(mlir_type ty) {
        if (auto ref = mlir::dyn_cast< hl::LValueType >(ty)) {
            return ref.getElementType();
        }

        return ty;
    }

    template< typename Op >
    operation default_stmt_visitor::visit_underlying_type_preserving_unary_op(
        const clang::UnaryOperator *op
    ) {
        if (auto arg = self.visit(op->getSubExpr())) {
            auto type = arg->getResult(0).getType();
            return bld.compose< Op >()
                .bind(self.location(op))
                .bind_transform(type, strip_lvalue)
                .bind(arg->getResult(0))
                .freeze();
        }

        return {};
    }

    template< typename Op >
    operation default_stmt_visitor::visit_cast_op(const clang::CastExpr *cast) {
        if (auto arg = self.visit(cast->getSubExpr())) {
            return bld.compose< Op >()
                .bind(self.location(cast))
                .bind(cast_result_type(cast, arg->getResultTypes().front()))
                .bind_transform(arg, first_result)
                .bind(cast_kind(cast))
                .freeze();
        }

        return {};
    }

} // namespace vast::cg
