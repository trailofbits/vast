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

    operation default_stmt_visitor::VisitCompoundStmt(const clang::CompoundStmt *stmt) {
        auto gen = mk_scoped_generator< block_generator >(self.scope, bld, self);
        return gen.emit(stmt);
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
    // Cast Operations
    //
    operation default_stmt_visitor::VisitImplicitCastExpr(const clang::ImplicitCastExpr *cast) {
        return visit_cast_op< hl::ImplicitCastOp >(cast);
    }

    operation default_stmt_visitor::VisitCStyleCastExpr(const clang::CStyleCastExpr *cast) {
        return visit_cast_op< hl::CStyleCastOp >(cast);
    }

    operation default_stmt_visitor::VisitBuiltinBitCastExpr(const clang::BuiltinBitCastExpr *cast) {
        return visit_cast_op< hl::BuiltinBitCastOp >(cast);
    }

    operation default_stmt_visitor::VisitCXXFunctionalCastExpr(const clang::CXXFunctionalCastExpr *cast) {
        return {};
    }

    operation default_stmt_visitor::VisitCXXConstCastExpr(const clang::CXXConstCastExpr *cast) {
        return {};
    }

    operation default_stmt_visitor::VisitCXXDynamicCastExpr(const clang::CXXDynamicCastExpr *cast) {
        return {};
    }

    operation default_stmt_visitor::VisitCXXReinterpretCastExpr(const clang::CXXReinterpretCastExpr *cast) {
        return {};
    }

    operation default_stmt_visitor::VisitCXXStaticCastExpr(const clang::CXXStaticCastExpr *cast) {
        return {};
    }

    mlir_type default_stmt_visitor::cast_result_type(const clang::CastExpr *cast, mlir_type from) {
        auto to_rvalue_cast     = [&] { return self.visit(cast->getType()); };
        auto lvalue_cast        = [&] { return self.visit_as_lvalue_type(cast->getType()); };
        auto non_lvalue_cast    = [&] { return self.visit(cast->getType()); };
        auto array_to_ptr_cast  = [&] { return self.visit(cast->getType()); };
        auto keep_category_cast = [&] {
            if (mlir::isa< hl::LValueType >(from))
                return lvalue_cast();
            return non_lvalue_cast();
        };

        switch (cast->getCastKind()) {
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
            // case clang::CastKind::CK_BaseToderived_tMemberPointer: return;
            // case clang::CastKind::CK_derived_tToBaseMemberPointer: return;
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
            default: return {};
        }
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

    //
    // Other Statements
    //
    operation default_stmt_visitor::VisitDeclRefExpr(const clang::DeclRefExpr *expr) {
        auto underlying = expr->getDecl()->getUnderlyingDecl();

        return llvm::TypeSwitch< const clang::NamedDecl *, operation >(underlying)
            .Case< clang::EnumDecl >([&] (auto /* e */) { return visit_enum_decl_ref(expr); })
            .Case< clang::VarDecl  >([&] (auto v) {
                if (v->isFileVarDecl()) {
                    return visit_file_var_decl_ref(expr);
                } else {
                    return visit_var_decl_ref(expr);
                }
            })
            .Case< clang::FunctionDecl >([&] (auto) { return visit_function_decl_ref(expr); })
            .Default([](const clang::NamedDecl *) { return operation{}; });
    }

    operation default_stmt_visitor::visit_enum_decl_ref(const clang::DeclRefExpr *expr) {
        return bld.compose< hl::EnumRefOp >()
            .bind(self.location(expr))
            .bind(self.visit(expr->getType()))
            .bind(self.symbol(expr))
            .freeze();
    }

    operation default_stmt_visitor::visit_file_var_decl_ref(const clang::DeclRefExpr *expr) {
        return bld.compose< hl::GlobalRefOp >()
            .bind(self.location(expr))
            .bind(self.visit_as_lvalue_type(expr->getType()))
            .bind(self.symbol(expr))
            .freeze();
    }

    operation default_stmt_visitor::visit_var_decl_ref(const clang::DeclRefExpr *expr) {
        if (auto name = self.symbol(expr)) {
            return bld.compose< hl::DeclRefOp >()
                .bind(self.location(expr))
                .bind(self.visit_as_lvalue_type(expr->getType()))
                .bind(self.scope.lookup_var(name.value()))
                .freeze();
        }

        return {};
    }

    operation default_stmt_visitor::visit_function_decl_ref(const clang::DeclRefExpr *expr) {
        return bld.compose< hl::FuncRefOp >()
            .bind(self.location(expr))
            .bind(self.visit_as_lvalue_type(expr->getType()))
            .bind(self.symbol(expr))
            .freeze();
    }

    hl::IdentKind ident_kind(const clang::PredefinedExpr *expr) {
        switch (expr->getIdentKind()) {
            case clang::PredefinedIdentKind::Func:
                return hl::IdentKind::Func;
            case clang::PredefinedIdentKind::Function:
                return hl::IdentKind::Function;
            case clang::PredefinedIdentKind::LFunction:
                return hl::IdentKind::LFunction;
            case clang::PredefinedIdentKind::FuncDName:
                return hl::IdentKind::FuncDName;
            case clang::PredefinedIdentKind::FuncSig:
                return hl::IdentKind::FuncSig;
            case clang::PredefinedIdentKind::LFuncSig:
                return hl::IdentKind::LFuncSig;
            case clang::PredefinedIdentKind::PrettyFunction:
                return hl::IdentKind::PrettyFunction;
            case clang::PredefinedIdentKind::PrettyFunctionNoVirtual:
                return hl::IdentKind::PrettyFunctionNoVirtual;
        }
    }

    operation default_stmt_visitor::VisitPredefinedExpr(const clang::PredefinedExpr *expr) {
        auto name = expr->getFunctionName();
        if (!name) {
            return {}; // unsupported clang::PredefinedExpr without name
        }

        return bld.compose< hl::PredefinedExpr >()
            .bind(self.location(expr))
            .bind(self.visit(expr->getType()))
            .bind_transform(self.visit(name), first_result)
            .bind(ident_kind(expr))
            .freeze();
    }

    operation default_stmt_visitor::VisitBreakStmt(const clang::BreakStmt *stmt) {
        return bld.compose< hl::BreakOp >().bind(self.location(stmt)).freeze();
    }

    operation default_stmt_visitor::VisitContinueStmt(const clang::ContinueStmt *stmt) {
        return bld.compose< hl::ContinueOp >().bind(self.location(stmt)).freeze();
    }

    operation default_stmt_visitor::VisitCaseStmt(const clang::CaseStmt *stmt) { return {}; }
    operation default_stmt_visitor::VisitDefaultStmt(const clang::DefaultStmt *stmt) { return {}; }
    operation default_stmt_visitor::VisitSwitchStmt(const clang::SwitchStmt *stmt) { return {}; }
    operation default_stmt_visitor::VisitDoStmt(const clang::DoStmt *stmt) { return {}; }
    // operation default_stmt_visitor::VisitCXXCatchStmt(const clang::CXXCatchStmt *stmt)
    // operation default_stmt_visitor::VisitCXXForRangeStmt(const clang::CXXForRangeStmt *stmt)
    // operation default_stmt_visitor::VisitCXXTryStmt(const clang::CXXTryStmt *stmt)
    // operation default_stmt_visitor::VisitCXXTryStmt(const clang::CXXTryStmt *stmt)
    // operation default_stmt_visitor::VisitCapturedStmt(const clang::CapturedStmt *stmt)
    operation default_stmt_visitor::VisitWhileStmt(const clang::WhileStmt *stmt) { return {}; }
    operation default_stmt_visitor::VisitForStmt(const clang::ForStmt *stmt) { return {}; }

    operation default_stmt_visitor::VisitGotoStmt(const clang::GotoStmt *stmt) {
        return bld.compose< hl::GotoStmt >()
            .bind(self.location(stmt))
            .bind_transform(self.visit(stmt->getLabel()), first_result)
            .freeze();
    }

    operation default_stmt_visitor::VisitLabelStmt(const clang::LabelStmt *stmt) { return {}; }
    operation default_stmt_visitor::VisitIfStmt(const clang::IfStmt *stmt) { return {}; }

    //
    // Expressions
    //
    operation default_stmt_visitor::VisitDeclStmt(const clang::DeclStmt *stmt) {
        if (stmt->isSingleDecl()) {
            return self.visit(stmt->getSingleDecl());
        } else {
            return {};
        }
    }
    operation default_stmt_visitor::VisitMemberExpr(const clang::MemberExpr *expr) { return {}; }

    operation default_stmt_visitor::VisitConditionalOperator(const clang::ConditionalOperator *op) { return {}; }

    operation default_stmt_visitor::VisitAddrLabelExpr(const clang::AddrLabelExpr *expr) {
        return bld.compose< hl::AddrLabelExpr >()
            .bind(self.location(expr))
            .bind(self.visit_as_lvalue_type(expr->getType()))
            .bind_transform(self.visit(expr->getLabel()), first_result)
            .freeze();
    }

    operation default_stmt_visitor::VisitConstantExpr(const clang::ConstantExpr *expr) {
        // TODO create ConstantExprOp
        return self.visit(expr->getSubExpr());
    }

    operation default_stmt_visitor::VisitArraySubscriptExpr(const clang::ArraySubscriptExpr *expr) {
        return bld.compose< hl::SubscriptOp >()
            .bind(self.location(expr))
            .bind(self.visit_as_lvalue_type(expr->getType()))
            .bind_transform(self.visit(expr->getBase()), first_result)
            .bind_transform(self.visit(expr->getIdx()), first_result)
            .freeze();
    }

    // operation default_stmt_visitor::VisitArrayTypeTraitExpr(const clang::ArrayTypeTraitExpr *expr)
    // operation default_stmt_visitor::VisitAsTypeExpr(const clang::AsTypeExpr *expr)
    // operation default_stmt_visitor::VisitAtomicExpr(const clang::AtomicExpr *expr)
    // operation default_stmt_visitor::VisitBlockExpr(const clang::BlockExpr *expr)

    // operation default_stmt_visitor::VisitCXXBindTemporaryExpr(const clang::CXXBindTemporaryExpr *expr) { return {}; }

    operation default_stmt_visitor::VisitCXXBoolLiteralExpr(const clang::CXXBoolLiteralExpr *expr) {
        return bld.compose< hl::ConstantOp >()
            .bind(self.location(expr))
            .bind(self.visit(expr->getType()))
            .bind(expr->getValue())
            .freeze();
    }

    // operation default_stmt_visitor::VisitCXXConstructExpr(const clang::CXXConstructExpr *expr)
    // operation default_stmt_visitor::VisitCXXTemporaryObjectExpr(const clang::CXXTemporaryObjectExpr *expr)
    // operation default_stmt_visitor::VisitCXXDefaultArgExpr(const clang::CXXDefaultArgExpr *expr)
    // operation default_stmt_visitor::VisitCXXDefaultInitExpr(const clang::CXXDefaultInitExpr *expr)
    // operation default_stmt_visitor::VisitCXXDeleteExpr(const clang::CXXDeleteExpr *expr)
    // operation default_stmt_visitor::VisitCXXDependentScopeMemberExpr(const clang::CXXDependentScopeMemberExpr *expr)
    // operation default_stmt_visitor::VisitCXXNewExpr(const clang::CXXNewExpr *expr)
    // operation default_stmt_visitor::VisitCXXNoexceptExpr(const clang::CXXNoexceptExpr *expr)
    // operation default_stmt_visitor::VisitCXXNullPtrLiteralExpr(const clang::CXXNullPtrLiteralExpr *expr)
    // operation default_stmt_visitor::VisitCXXPseudoDestructorExpr(const clang::CXXPseudoDestructorExpr *expr)
    // operation default_stmt_visitor::VisitCXXScalarValueInitExpr(const clang::CXXScalarValueInitExpr *expr)
    // operation default_stmt_visitor::VisitCXXStdInitializerListExpr(const clang::CXXStdInitializerListExpr *expr)
    // operation default_stmt_visitor::VisitCXXThisExpr(const clang::CXXThisExpr *expr)
    // operation default_stmt_visitor::VisitCXXThrowExpr(const clang::CXXThrowExpr *expr)
    // operation default_stmt_visitor::VisitCXXTypeidExpr(const clang::CXXTypeidExpr *expr)
    // operation CXXFoldExpr(const clang::CXXFoldExpr *expr)
    // operation default_stmt_visitor::VisitCXXUnresolvedConstructExpr(const clang::CXXThrowExpr *expr)
    // operation default_stmt_visitor::VisitCXXUuidofExpr(const clang::CXXUuidofExpr *expr)

    operation default_stmt_visitor::VisitCallExpr(const clang::CallExpr *expr) { return {}; }

    // operation default_stmt_visitor::VisitCXXMemberCallExpr(const clang::CXXMemberCallExpr *expr)
    // operation default_stmt_visitor::VisitCXXOperatorCallExpr(const clang::CXXOperatorCallExpr *expr)

    // operation default_stmt_visitor::VisitOffsetOfExpr(const clang::OffsetOfExpr *expr)
    // operation default_stmt_visitor::VisitOpaqueValueExpr(const clang::OpaqueValueExpr *expr)
    // operation default_stmt_visitor::VisitOverloadExpr(const clang::OverloadExpr *expr)

    operation default_stmt_visitor::VisitParenExpr(const clang::ParenExpr *expr) { return {}; }

    // operation default_stmt_visitor::VisitParenListExpr(const clang::ParenListExpr *expr)
    operation default_stmt_visitor::VisitStmtExpr(const clang::StmtExpr *expr) { return {}; }

    operation default_stmt_visitor::VisitUnaryExprOrTypeTraitExpr(const clang::UnaryExprOrTypeTraitExpr *expr) { return {}; }

    operation default_stmt_visitor::VisitVAArgExpr(const clang::VAArgExpr *expr) {
        return bld.compose< hl::VAArgExpr >()
            .bind(self.location(expr))
            .bind(self.visit(expr->getType()))
            .bind_transform(self.visit(expr->getSubExpr()), results)
            .freeze();
    }

    operation default_stmt_visitor::VisitNullStmt(const clang::NullStmt *stmt) {
        return bld.compose< hl::SkipStmt >().bind(self.location(stmt)).freeze();
    }

    operation default_stmt_visitor::VisitCXXThisExpr(const clang::CXXThisExpr *expr) {
        return bld.compose< hl::ThisOp >()
            .bind(self.location(expr))
            .bind(self.visit(expr->getType()))
            .freeze();
    }

} // namespace vast::cg
