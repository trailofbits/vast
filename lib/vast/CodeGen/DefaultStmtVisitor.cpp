// Copyright (c) 2022-present, Trail of Bits, Inc.

#include "vast/CodeGen/DefaultStmtVisitor.hpp"

#include "vast/CodeGen/CodeGenBlock.hpp"
#include "vast/CodeGen/CodeGenVisitorBase.hpp"

#include "vast/Dialect/HighLevel/HighLevelOps.hpp"

namespace vast::cg
{
    using hl::CastKind;

    CastKind cast_kind(const clang::CastExpr *expr)
    {
        switch (expr->getCastKind()) {
            case clang::CastKind::CK_Dependent: return CastKind::Dependent;
            case clang::CastKind::CK_BitCast: return CastKind::BitCast;
            case clang::CastKind::CK_LValueBitCast: return CastKind::LValueBitCast;
            case clang::CastKind::CK_LValueToRValueBitCast: return CastKind::LValueToRValueBitCast;
            case clang::CastKind::CK_LValueToRValue: return CastKind::LValueToRValue;
            case clang::CastKind::CK_NoOp: return CastKind::NoOp;

            case clang::CastKind::CK_BaseToDerived: return CastKind::BaseToDerived;
            case clang::CastKind::CK_DerivedToBase: return CastKind::DerivedToBase;
            case clang::CastKind::CK_UncheckedDerivedToBase: return CastKind::UncheckedDerivedToBase;
            case clang::CastKind::CK_Dynamic: return CastKind::Dynamic;
            case clang::CastKind::CK_ToUnion: return CastKind::ToUnion;

            case clang::CastKind::CK_ArrayToPointerDecay: return CastKind::ArrayToPointerDecay;
            case clang::CastKind::CK_FunctionToPointerDecay: return CastKind::FunctionToPointerDecay;
            case clang::CastKind::CK_NullToPointer: return CastKind::NullToPointer;
            case clang::CastKind::CK_NullToMemberPointer: return CastKind::NullToMemberPointer;
            case clang::CastKind::CK_BaseToDerivedMemberPointer: return CastKind::BaseToDerivedMemberPointer;
            case clang::CastKind::CK_DerivedToBaseMemberPointer: return CastKind::DerivedToBaseMemberPointer;
            case clang::CastKind::CK_MemberPointerToBoolean: return CastKind::MemberPointerToBoolean;
            case clang::CastKind::CK_ReinterpretMemberPointer: return CastKind::ReinterpretMemberPointer;
            case clang::CastKind::CK_UserDefinedConversion: return CastKind::UserDefinedConversion;
            case clang::CastKind::CK_ConstructorConversion: return CastKind::ConstructorConversion;

            case clang::CastKind::CK_IntegralToPointer: return CastKind::IntegralToPointer;
            case clang::CastKind::CK_PointerToIntegral: return CastKind::PointerToIntegral;
            case clang::CastKind::CK_PointerToBoolean : return CastKind::PointerToBoolean;

            case clang::CastKind::CK_ToVoid: return CastKind::ToVoid;
            case clang::CastKind::CK_VectorSplat: return CastKind::VectorSplat;

            case clang::CastKind::CK_IntegralCast: return CastKind::IntegralCast;
            case clang::CastKind::CK_IntegralToBoolean: return CastKind::IntegralToBoolean;
            case clang::CastKind::CK_IntegralToFloating: return CastKind::IntegralToFloating;
            case clang::CastKind::CK_FloatingToFixedPoint: return CastKind::FloatingToFixedPoint;
            case clang::CastKind::CK_FixedPointToFloating: return CastKind::FixedPointToFloating;
            case clang::CastKind::CK_FixedPointCast: return CastKind::FixedPointCast;
            case clang::CastKind::CK_FixedPointToIntegral: return CastKind::FixedPointToIntegral;
            case clang::CastKind::CK_IntegralToFixedPoint: return CastKind::IntegralToFixedPoint;
            case clang::CastKind::CK_FixedPointToBoolean: return CastKind::FixedPointToBoolean;
            case clang::CastKind::CK_FloatingToIntegral: return CastKind::FloatingToIntegral;
            case clang::CastKind::CK_FloatingToBoolean: return CastKind::FloatingToBoolean;
            case clang::CastKind::CK_BooleanToSignedIntegral: return CastKind::BooleanToSignedIntegral;
            case clang::CastKind::CK_FloatingCast: return CastKind::FloatingCast;

            case clang::CastKind::CK_CPointerToObjCPointerCast: return CastKind::CPointerToObjCPointerCast;
            case clang::CastKind::CK_BlockPointerToObjCPointerCast: return CastKind::BlockPointerToObjCPointerCast;
            case clang::CastKind::CK_AnyPointerToBlockPointerCast: return CastKind::AnyPointerToBlockPointerCast;
            case clang::CastKind::CK_ObjCObjectLValueCast: return CastKind::ObjCObjectLValueCast;

            case clang::CastKind::CK_FloatingRealToComplex: return CastKind::FloatingRealToComplex;
            case clang::CastKind::CK_FloatingComplexToReal: return CastKind::FloatingComplexToReal;
            case clang::CastKind::CK_FloatingComplexToBoolean: return CastKind::FloatingComplexToBoolean;
            case clang::CastKind::CK_FloatingComplexCast: return CastKind::FloatingComplexCast;
            case clang::CastKind::CK_FloatingComplexToIntegralComplex: return CastKind::FloatingComplexToIntegralComplex;
            case clang::CastKind::CK_IntegralRealToComplex: return CastKind::IntegralRealToComplex;
            case clang::CastKind::CK_IntegralComplexToReal: return CastKind::IntegralComplexToReal;
            case clang::CastKind::CK_IntegralComplexToBoolean: return CastKind::IntegralComplexToBoolean;
            case clang::CastKind::CK_IntegralComplexCast: return CastKind::IntegralComplexCast;
            case clang::CastKind::CK_IntegralComplexToFloatingComplex: return CastKind::IntegralComplexToFloatingComplex;

            case clang::CastKind::CK_ARCProduceObject: return CastKind::ARCProduceObject;
            case clang::CastKind::CK_ARCConsumeObject: return CastKind::ARCConsumeObject;
            case clang::CastKind::CK_ARCReclaimReturnedObject: return CastKind::ARCReclaimReturnedObject;
            case clang::CastKind::CK_ARCExtendBlockObject: return CastKind::ARCExtendBlockObject;

            case clang::CastKind::CK_AtomicToNonAtomic: return CastKind::AtomicToNonAtomic;
            case clang::CastKind::CK_NonAtomicToAtomic: return CastKind::NonAtomicToAtomic;

            case clang::CastKind::CK_CopyAndAutoreleaseBlockObject: return CastKind::CopyAndAutoreleaseBlockObject;
            case clang::CastKind::CK_BuiltinFnToFnPtr: return CastKind::BuiltinFnToFnPtr;

            case clang::CastKind::CK_ZeroToOCLOpaqueType: return CastKind::ZeroToOCLOpaqueType;
            case clang::CastKind::CK_AddressSpaceConversion: return CastKind::AddressSpaceConversion;
            case clang::CastKind::CK_IntToOCLSampler: return CastKind::IntToOCLSampler;

            case clang::CastKind::CK_MatrixCast: return CastKind::MatrixCast;
        }

        VAST_UNIMPLEMENTED_MSG( "unsupported cast kind" );
    }

    using hl::IdentKind;

    IdentKind ident_kind(const clang::PredefinedExpr *expr)
    {
        switch(expr->getIdentKind())
        {
            case clang::PredefinedIdentKind::Func : return IdentKind::Func;
            case clang::PredefinedIdentKind::Function : return IdentKind::Function;
            case clang::PredefinedIdentKind::LFunction : return IdentKind::LFunction;
            case clang::PredefinedIdentKind::FuncDName : return IdentKind::FuncDName;
            case clang::PredefinedIdentKind::FuncSig : return IdentKind::FuncSig;
            case clang::PredefinedIdentKind::LFuncSig : return IdentKind::LFuncSig;
            case clang::PredefinedIdentKind::PrettyFunction :
                return IdentKind::PrettyFunction;
            case clang::PredefinedIdentKind::PrettyFunctionNoVirtual :
                return IdentKind::PrettyFunctionNoVirtual;
        }
    }

    //
    // Binary Operations
    //

    operation default_stmt_visitor::VisitBinPtrMemD(const clang::BinaryOperator */* op */) {
        return {};
    }

    operation default_stmt_visitor::VisitBinPtrMemI(const clang::BinaryOperator */* op */) {
        return {};
    }

    operation default_stmt_visitor::VisitBinMul(const clang::BinaryOperator *op) {
        return visit_ifbin_op< hl::MulIOp, hl::MulIOp, hl::MulFOp >(op);
    }

    operation default_stmt_visitor::VisitBinDiv(const clang::BinaryOperator *op) {
        return visit_ifbin_op< hl::DivUOp, hl::DivSOp, hl::DivFOp >(op);
    }

    operation default_stmt_visitor::VisitBinRem(const clang::BinaryOperator *op) {
        return visit_ifbin_op< hl::RemUOp, hl::RemSOp, hl::RemFOp >(op);
    }

    operation default_stmt_visitor::VisitBinAdd(const clang::BinaryOperator *op) {
        return visit_ifbin_op< hl::AddIOp, hl::AddIOp, hl::AddFOp >(op);
    }

    operation default_stmt_visitor::VisitBinSub(const clang::BinaryOperator *op) {
        return visit_ifbin_op< hl::SubIOp, hl::SubIOp, hl::SubFOp >(op);
    }

    operation default_stmt_visitor::VisitBinShl(const clang::BinaryOperator *op) {
        return visit_bin_op< hl::BinShlOp >(op);
    }

    operation default_stmt_visitor::VisitBinShr(const clang::BinaryOperator *op) {
        return visit_ibin_op< hl::BinLShrOp, hl::BinAShrOp >(op);
    }

    using ipred = hl::Predicate;
    using fpred = hl::FPredicate;

    operation default_stmt_visitor::VisitBinLT(const clang::BinaryOperator *op) {
        return visit_cmp_op< ipred::ult, ipred::slt, fpred::olt >(op);
    }

    operation default_stmt_visitor::VisitBinGT(const clang::BinaryOperator *op) {
        return visit_cmp_op< ipred::ugt, ipred::sgt, fpred::ogt >(op);
    }

    operation default_stmt_visitor::VisitBinLE(const clang::BinaryOperator *op) {
        return visit_cmp_op< ipred::ule, ipred::sle, fpred::ole >(op);
    }

    operation default_stmt_visitor::VisitBinGE(const clang::BinaryOperator *op) {
        return visit_cmp_op< ipred::uge, ipred::sge, fpred::oge >(op);
    }

    operation default_stmt_visitor::VisitBinEQ(const clang::BinaryOperator *op) {
        return visit_cmp_op< ipred::eq, ipred::eq, fpred::oeq >(op);
    }

    operation default_stmt_visitor::VisitBinNE(const clang::BinaryOperator *op) {
        return visit_cmp_op< ipred::ne, ipred::ne, fpred::one >(op);
    }

    operation default_stmt_visitor::VisitBinAnd(const clang::BinaryOperator *op) {
        return visit_bin_op< hl::BinAndOp >(op);
    }

    operation default_stmt_visitor::VisitBinXor(const clang::BinaryOperator *op) {
        return visit_bin_op< hl::BinXorOp >(op);
    }

    operation default_stmt_visitor::VisitBinOr(const clang::BinaryOperator *op) {
        return visit_bin_op< hl::BinOrOp >(op);
    }

    //
    // Compound Assignment Operations
    //
    operation default_stmt_visitor::VisinBinAssign(const clang::CompoundAssignOperator *op) {
        return visit_assign_bin_op< hl::AssignOp >(op);
    }

    operation default_stmt_visitor::VisitBinMulAssign(const clang::CompoundAssignOperator *op) {
        return visit_assign_ifbin_op< hl::MulIAssignOp, hl::MulIAssignOp, hl::MulFAssignOp >(op);
    }

    operation default_stmt_visitor::VisitBinDivAssign(const clang::CompoundAssignOperator *op) {
        return visit_assign_ifbin_op< hl::DivUAssignOp, hl::DivSAssignOp, hl::DivFAssignOp >(op);
    }

    operation default_stmt_visitor::VisitBinRemAssign(const clang::CompoundAssignOperator *op) {
        return visit_assign_ibin_op< hl::RemUAssignOp, hl::RemSAssignOp >(op);
    }

    operation default_stmt_visitor::VisitBinAddAssign(const clang::CompoundAssignOperator *op) {
        return visit_assign_ifbin_op< hl::AddIAssignOp, hl::AddIAssignOp, hl::AddFAssignOp >(op);
    }

    operation default_stmt_visitor::VisitBinSubAssign(const clang::CompoundAssignOperator *op) {
        return visit_assign_ifbin_op< hl::SubIAssignOp, hl::SubIAssignOp, hl::SubFAssignOp >(op);
    }

    operation default_stmt_visitor::VisitBinShlAssign(const clang::CompoundAssignOperator *op) {
        return visit_assign_bin_op< hl::BinShlAssignOp >(op);
    }

    operation default_stmt_visitor::VisitBinShrAssign(const clang::CompoundAssignOperator *op) {
        return visit_assign_ibin_op< hl::BinLShrAssignOp, hl::BinAShrAssignOp >(op);
    }

    operation default_stmt_visitor::VisitBinAndAssign(const clang::CompoundAssignOperator *op) {
        return visit_assign_bin_op< hl::BinAndAssignOp >(op);
    }

    operation default_stmt_visitor::VisitBinXorAssign(const clang::CompoundAssignOperator *op) {
        return visit_assign_bin_op< hl::BinXorAssignOp >(op);
    }

    operation default_stmt_visitor::VisitBinOrAssign(const clang::CompoundAssignOperator *op) {
        return visit_assign_bin_op< hl::BinOrAssignOp >(op);
    }

    operation default_stmt_visitor::VisitBinComma(const clang::BinaryOperator *op) {
        return bld.compose< hl::BinComma >()
            .bind(self.location(op))
            .bind(self.visit(op->getType()))
            .bind_transform(self.visit(op->getLHS()), first_result)
            .bind_transform(self.visit(op->getRHS()), first_result)
            .freeze();
    }

    //
    // Unary Operations
    //
    operation default_stmt_visitor::VisitUnaryPostInc(const clang::UnaryOperator *op) {
        return visit_underlying_type_preserving_unary_op< hl::PostIncOp >(op);
    }

    operation default_stmt_visitor::VisitUnaryPostDec(const clang::UnaryOperator *op) {
        return visit_underlying_type_preserving_unary_op< hl::PostDecOp >(op);
    }

    operation default_stmt_visitor::VisitUnaryPreInc(const clang::UnaryOperator *op) {
        return visit_underlying_type_preserving_unary_op< hl::PreIncOp >(op);
    }

    operation default_stmt_visitor::VisitUnaryPreDec(const clang::UnaryOperator *op) {
        return visit_underlying_type_preserving_unary_op< hl::PreDecOp >(op);
    }

    operation default_stmt_visitor::VisitUnaryAddrOf(const clang::UnaryOperator *op) {
        return visit_unary_op< hl::AddressOf >(op, self.visit(op->getType()));
    }

    operation default_stmt_visitor::VisitUnaryDeref(const clang::UnaryOperator *op) {
        return visit_unary_op< hl::Deref >(op, self.visit_as_lvalue_type(op->getType()));
    }

    operation default_stmt_visitor::VisitUnaryPlus(const clang::UnaryOperator *op) {
        return visit_underlying_type_preserving_unary_op< hl::PlusOp >(op);
    }

    operation default_stmt_visitor::VisitUnaryMinus(const clang::UnaryOperator *op) {
        return visit_underlying_type_preserving_unary_op< hl::MinusOp >(op);
    }

    operation default_stmt_visitor::VisitUnaryNot(const clang::UnaryOperator *op) {
        return visit_underlying_type_preserving_unary_op< hl::NotOp >(op);
    }

    operation default_stmt_visitor::VisitUnaryLNot(const clang::UnaryOperator *op) {
        return visit_unary_op< hl::LNotOp >(op, self.visit(op->getType()));
    }

    operation default_stmt_visitor::VisitUnaryExtension(const clang::UnaryOperator *op) {
        return visit_underlying_type_preserving_unary_op< hl::ExtensionOp >(op);
    }
    operation default_stmt_visitor::VisitUnaryReal(const clang::UnaryOperator *op) {
        return {};
    }

    operation default_stmt_visitor::VisitUnaryImag(const clang::UnaryOperator *op) {
        return {};
    }

    operation default_stmt_visitor::VisitUnaryCoawait(const clang::UnaryOperator *op) {
        return {};
    }

    //
    // ControlFlow Statements
    //
    operation default_stmt_visitor::VisitReturnStmt(const clang::ReturnStmt *stmt) {
        auto loc = self.location(stmt);
        auto op = bld.compose< hl::ReturnOp >().bind(loc);

        if (stmt->getRetValue()) {
            return std::move(op).bind(self.visit(stmt->getRetValue())->getResults()).freeze();
        } else {
            return std::move(op).bind(bld.void_value(loc)).freeze();
        }
    }

    //
    // Literals
    //
    operation default_stmt_visitor::VisistCharacterLiteral(const clang::CharacterLiteral *lit) {
        return bld.constant(self.location(lit), self.visit(lit->getType()), lit->getValue()).getDefiningOp();
    }

    operation default_stmt_visitor::VisitIntegerLiteral(const clang::IntegerLiteral *lit) {
        return bld.constant(self.location(lit), self.visit(lit->getType()), llvm::APSInt(lit->getValue(), false)).getDefiningOp();
    }

    operation default_stmt_visitor::VisitFloatingLiteral(const clang::FloatingLiteral *lit) {
        return bld.constant(self.location(lit), self.visit(lit->getType()), lit->getValue()).getDefiningOp();
    }

    operation default_stmt_visitor::VisitStringLiteral(const clang::StringLiteral *lit) {
        VAST_ASSERT(lit->isLValue() && "string literal is expected to be an lvalue");
        return bld.constant(self.location(lit), self.visit_as_lvalue_type(lit->getType()), lit->getString()).getDefiningOp();
    }

    operation default_stmt_visitor::VisitUserDefinedLiteral(const clang::UserDefinedLiteral */* lit */) {
        return {};
    }

    operation default_stmt_visitor::VisitCompoundLiteralExpr(const clang::CompoundLiteralExpr */* lit */) {
        return {};
    }

    operation default_stmt_visitor::VisitFixedPointLiteral(const clang::FixedPointLiteral */* lit */) {
        return {};
    }

} // namespace vast::cg
