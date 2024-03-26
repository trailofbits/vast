// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <clang/AST/StmtVisitor.h>
#include <clang/AST/OperationKinds.h>
VAST_UNRELAX_WARNINGS

#include "vast/CodeGen/CodeGenVisitorBase.hpp"

namespace vast::cg {

    struct default_stmt_visitor : stmt_visitor_base< default_stmt_visitor >
    {
        using base = stmt_visitor_base< default_stmt_visitor >;

        explicit default_stmt_visitor(codegen_builder &bld, visitor_view self)
            : base(bld, self)
        {}

        operation visit(const clang_stmt *stmt) { return Visit(stmt); }

        using base::Visit;

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

        //
        // Compound Assignment Operations
        //
        template< typename Op >
        operation visit_assign_bin_op(const clang::CompoundAssignOperator *op);

        template< typename UOp, typename SOp >
        operation visit_assign_ibin_op(const clang::CompoundAssignOperator *op);

        template< typename UOp, typename SOp, typename FOp >
        operation visit_assign_ifbin_op(const clang::CompoundAssignOperator *op);

        operation VisinBinAssign(const clang::CompoundAssignOperator *op);

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
        // ControlFlow Statements
        //
        operation VisitReturnStmt(const clang::ReturnStmt *stmt);

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
    };

    static inline auto first_result = [] (auto op) {
        return op->getResult(0);
    };

    template< typename Op >
    operation default_stmt_visitor::visit_bin_op(const clang::BinaryOperator *op) {
        auto lhs = self.visit(op->getLHS());
        auto rhs = self.visit(op->getRHS());

        return bld.compose< Op >()
            .bind(self.location(op))
            .bind(self.visit(op->getType()))
            .bind_transform(lhs, first_result)
            .bind_transform(rhs, first_result)
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
        auto lhs = self.visit(op->getLHS());
        auto rhs = self.visit(op->getRHS());

        return bld.compose< hl::CmpOp >()
            .bind(self.location(op))
            .bind(self.visit(op->getType()))
            .bind(pred)
            .bind_transform(lhs, first_result)
            .bind_transform(rhs, first_result)
            .freeze();
    }

    template< hl::FPredicate pred >
    operation default_stmt_visitor::visit_fcmp_op(const clang::BinaryOperator *op) {
        auto lhs = self.visit(op->getLHS());
        auto rhs = self.visit(op->getRHS());

        return bld.compose< hl::FCmpOp >()
            .bind(self.location(op))
            .bind(self.visit(op->getType()))
            .bind(pred)
            .bind_transform(lhs, first_result)
            .bind_transform(rhs, first_result)
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

    template< typename Op >
    operation default_stmt_visitor::visit_assign_bin_op(const clang::CompoundAssignOperator *op) {
        auto lhs = self.visit(op->getLHS());
        auto rhs = self.visit(op->getRHS());

        return bld.compose< Op >()
            .bind(self.location(op))
            .bind_transform(lhs, first_result)
            .bind_transform(rhs, first_result)
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

} // namespace vast::cg
