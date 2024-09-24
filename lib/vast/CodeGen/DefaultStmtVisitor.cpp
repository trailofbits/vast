// Copyright (c) 2022-present, Trail of Bits, Inc.

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <clang/AST/ParentMapContext.h>
VAST_UNRELAX_WARNINGS

#include "vast/CodeGen/DefaultStmtVisitor.hpp"
#include "vast/CodeGen/DefaultTypeVisitor.hpp"

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

    operation default_stmt_visitor::VisitBinLAnd(const clang::BinaryOperator *op) {
        return visit_logical_op< hl::BinLAndOp >(op);
    }

    operation default_stmt_visitor::VisitBinLOr(const clang::BinaryOperator *op) {
        return visit_logical_op< hl::BinLOrOp >(op);
    }

    //
    // Compound Assignment Operations
    //
    operation default_stmt_visitor::VisitBinAssign(const clang::BinaryOperator *op) {
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
        return visit_unary_op< hl::Deref >(op, visit_as_lvalue_type(self, mctx, op->getType()));
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
        return visit_unary_op< hl::RealOp >(op, visit_maybe_lvalue_result_type(op));
    }

    operation default_stmt_visitor::VisitUnaryImag(const clang::UnaryOperator *op) {
        return visit_unary_op< hl::ImagOp >(op, visit_maybe_lvalue_result_type(op));
    }

    operation default_stmt_visitor::VisitUnaryCoawait(const clang::UnaryOperator *op) {
        return {};
    }

    //
    // Assembly Statements
    //
    operation default_stmt_visitor::VisitAsmStmt(const clang::AsmStmt *stmt) {
        return {};
    }

    operation default_stmt_visitor::VisitGCCAsmStmt(const clang::GCCAsmStmt *stmt) {
        auto get_string_attr = [&](mlir::StringRef str) {
            return mlir::StringAttr::get(&mctx, str);
        };

        auto asm_attr = get_string_attr(stmt->getAsmString()->getString());

        if (stmt->isSimple()) {
            return bld.compose< hl::AsmOp >()
                .bind(self.location(stmt))
                .bind_always(asm_attr)
                .bind_always(stmt->isVolatile())
                .bind_always(false /* has goto */)
                .freeze();
        }

        values_t outputs;
        values_t inputs;
        attrs_t  out_names;
        attrs_t  in_names;
        attrs_t  out_constraints;
        attrs_t  in_constraints;
        attrs_t  clobbers;
        values_t labels;

        auto get_integer_attr = [&](int i) {
            return bld.getI64IntegerAttr(i);
        };

        auto get_out_expr       = [&](int i) { return stmt->getOutputExpr(i); };
        auto get_out_name       = [&](int i) { return stmt->getOutputName(i); };
        auto get_out_constraint = [&](int i) { return stmt->getOutputConstraint(i); };

        auto get_in_expr       = [&](int i) { return stmt->getInputExpr(i); };
        auto get_in_name       = [&](int i) { return stmt->getInputName(i); };
        auto get_in_constraint = [&](int i) { return stmt->getInputConstraint(i); };

        int arg_num = 0;

        auto fill_vectors = [&](
            int size, const auto &get_expr, const auto &get_name,
            const auto &get_constraint, auto &vals, auto &names,
            auto &constraints
        ) {
            for (int i = 0; i < size; i++) {
                auto id = get_name(i);
                if (id.size()) {
                    names.push_back(get_string_attr(id));
                } else {
                    names.push_back(get_integer_attr(arg_num));
                }
                arg_num++;

                constraints.push_back(get_string_attr(get_constraint(i)));
                vals.emplace_back(visit(get_expr(i))->getResult(0));
            }
        };

        fill_vectors(
            stmt->getNumOutputs(), get_out_expr, get_out_name, get_out_constraint, outputs,
            out_names, out_constraints
        );

        fill_vectors(
            stmt->getNumInputs(), get_in_expr, get_in_name, get_in_constraint, inputs,
            in_names, in_constraints
        );

        if (stmt->isAsmGoto()) {
            for (const auto &lab : stmt->labels()) {
                labels.emplace_back(visit(lab)->getResult(0));
            }
        }

        for (size_t i = 0; i < stmt->getNumClobbers(); i++) {
            clobbers.emplace_back(get_string_attr(stmt->getClobber(i)));
        }

        auto get_array_attr = [&](attrs_t &arr) {
            return mlir::ArrayAttr::get(&mctx, mlir::ArrayRef(arr));
        };

        return bld.compose< hl::AsmOp >()
            .bind(self.location(stmt))
            .bind_always(asm_attr)
            .bind_always(stmt->isVolatile())
            .bind_always(stmt->isAsmGoto())
            .bind_always(outputs)
            .bind_always(inputs)
            .bind_always(get_array_attr(out_names))
            .bind_always(get_array_attr(in_names))
            .bind_always(get_array_attr(out_constraints))
            .bind_always(get_array_attr(in_constraints))
            .bind_always(get_array_attr(clobbers))
            .bind_always(labels)
            .freeze();
    }

    operation default_stmt_visitor::VisVisitMSAsmStmtitAsmStmt(const clang::MSAsmStmt *stmt) {
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
        auto lvalue_cast        = [&] { return visit_as_lvalue_type(self, mctx, cast->getType()); };
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

            case clang::CastKind::CK_AtomicToNonAtomic:
            case clang::CastKind::CK_NonAtomicToAtomic:
                return keep_category_cast();

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
        auto op = bld.compose< hl::ReturnOp >().bind(self.location(stmt));

        if (stmt->getRetValue()) {
            return std::move(op)
                .bind_transform(self.visit(stmt->getRetValue()), first_result)
                .freeze();
        } else {
            return std::move(op)
                .bind(bld.void_value(self.location(stmt).value()))
                .freeze();
        }
    }

    //
    // Literals
    //
    mlir_type default_stmt_visitor::visit_maybe_lvalue_result_type(const clang::Expr *lit){
        if (lit->isLValue())
            return visit_as_lvalue_type(self, mctx, lit->getType());
        return self.visit(lit->getType());
    }

    operation default_stmt_visitor::VisitCharacterLiteral(const clang::CharacterLiteral *lit) {
        return bld.constant(self.location(lit).value(), self.visit(lit->getType()), apsint(lit->getValue())).getDefiningOp();
    }

    operation default_stmt_visitor::VisitIntegerLiteral(const clang::IntegerLiteral *lit) {
        return bld.constant(self.location(lit).value(), self.visit(lit->getType()), llvm::APSInt(lit->getValue(), false)).getDefiningOp();
    }

    operation default_stmt_visitor::VisitFloatingLiteral(const clang::FloatingLiteral *lit) {
        return bld.constant(self.location(lit).value(), self.visit(lit->getType()), lit->getValue()).getDefiningOp();
    }

    operation default_stmt_visitor::VisitStringLiteral(const clang::StringLiteral *lit) {
        return bld.constant(self.location(lit).value(),
                            visit_maybe_lvalue_result_type(lit),
                            lit->getString()).getDefiningOp();
    }

    operation default_stmt_visitor::VisitUserDefinedLiteral(const clang::UserDefinedLiteral */* lit */) {
        return {};
    }

    operation default_stmt_visitor::VisitCompoundLiteralExpr(const clang::CompoundLiteralExpr *lit) {
        return bld.compose< hl::CompoundLiteralOp >()
            .bind(self.location(lit))
            .bind(visit_maybe_lvalue_result_type(lit))
            .bind_always(mk_value_builder(lit->getInitializer()))
            .freeze();
    }

    operation default_stmt_visitor::VisitFixedPointLiteral(const clang::FixedPointLiteral */* lit */) {
        return {};
    }

    operation default_stmt_visitor::VisitImaginaryLiteral(const clang::ImaginaryLiteral *lit) {
        return bld.compose< hl::InitializedConstantOp >()
            .bind(self.location(lit))
            .bind(self.visit(lit->getType()))
            .bind_always(mk_value_builder(lit->getSubExpr()))
            .freeze();
    }

    //
    // Other Statements
    //
    operation default_stmt_visitor::VisitDeclRefExpr(const clang::DeclRefExpr *expr) {
        auto underlying = expr->getDecl()->getUnderlyingDecl();
        return llvm::TypeSwitch< const clang::NamedDecl *, operation >(underlying)
            .Case< clang::EnumConstantDecl >([&] (auto /* e */) { return visit_enum_decl_ref(expr); })
            .Case< clang::VarDecl >([&] (auto v) {
                return visit_var_decl_ref(expr);
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

    operation default_stmt_visitor::visit_var_decl_ref(const clang::DeclRefExpr *expr) {
        if (auto name = self.symbol(expr)) {
            return bld.compose< hl::DeclRefOp >()
                .bind(self.location(expr))
                .bind(visit_as_lvalue_type(self, mctx, expr->getType()))
                .bind(self.symbol(expr))
                .freeze();
        }

        return {};
    }

    operation default_stmt_visitor::visit_function_decl_ref(const clang::DeclRefExpr *expr) {
        return bld.compose< hl::FuncRefOp >()
            .bind(self.location(expr))
            .bind(visit_maybe_lvalue_result_type(expr))
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
            .bind_always(ident_kind(expr))
            .freeze();
    }

    operation default_stmt_visitor::VisitBreakStmt(const clang::BreakStmt *stmt) {
        return bld.compose< hl::BreakOp >().bind(self.location(stmt)).freeze();
    }

    operation default_stmt_visitor::VisitContinueStmt(const clang::ContinueStmt *stmt) {
        return bld.compose< hl::ContinueOp >().bind(self.location(stmt)).freeze();
    }

    operation default_stmt_visitor::VisitCaseStmt(const clang::CaseStmt *stmt) {
        return bld.compose< hl::CaseOp >()
            .bind(self.location(stmt))
            .bind_always(mk_value_builder(stmt->getLHS()))
            .bind_always(mk_optional_region_builder(stmt->getSubStmt()))
            .freeze();
    }

    operation default_stmt_visitor::VisitDefaultStmt(const clang::DefaultStmt *stmt) {
        return bld.compose< hl::DefaultOp >()
            .bind(self.location(stmt))
            .bind_always(mk_optional_region_builder(stmt->getSubStmt()))
            .freeze();
    }

    operation default_stmt_visitor::VisitSwitchStmt(const clang::SwitchStmt *stmt) {
        if (stmt->getInit()) {
            return {};
        }

        return bld.compose< hl::SwitchOp >()
            .bind(self.location(stmt))
            .bind_always(mk_value_builder(stmt->getCond()))
            .bind_always(mk_optional_region_builder(stmt->getBody()))
            .freeze();
    }

    operation default_stmt_visitor::VisitDoStmt(const clang::DoStmt *stmt) {
        return bld.compose< hl::DoOp >()
            .bind(self.location(stmt))
            .bind_always(mk_optional_region_builder(stmt->getBody()))
            .bind_always(mk_cond_builder(stmt->getCond()))
            .freeze();
    }

    // operation default_stmt_visitor::VisitCXXCatchStmt(const clang::CXXCatchStmt *stmt)
    // operation default_stmt_visitor::VisitCXXForRangeStmt(const clang::CXXForRangeStmt *stmt)
    // operation default_stmt_visitor::VisitCXXTryStmt(const clang::CXXTryStmt *stmt)
    // operation default_stmt_visitor::VisitCXXTryStmt(const clang::CXXTryStmt *stmt)
    // operation default_stmt_visitor::VisitCapturedStmt(const clang::CapturedStmt *stmt)

    operation default_stmt_visitor::VisitWhileStmt(const clang::WhileStmt *stmt) {
        return bld.compose< hl::WhileOp >()
            .bind(self.location(stmt))
            .bind_always(mk_cond_builder(stmt->getCond()))
            .bind_always(mk_optional_region_builder(stmt->getBody()))
            .freeze();
    }

    operation default_stmt_visitor::VisitForStmt(const clang::ForStmt *stmt) {
        if (auto init = stmt->getInit()) {
            // FIXME: make initialization section
            self.visit(init);
        }

        return bld.compose< hl::ForOp >()
            .bind(self.location(stmt))
            .bind_choose(
                  stmt->getCond()
                , mk_cond_builder(stmt->getCond())
                , mk_true_yielder()
            )
            .bind_always(mk_optional_region_builder(stmt->getInc()))
            .bind_always(mk_optional_region_builder(stmt->getBody()))
            .freeze();
    }

    operation default_stmt_visitor::VisitGotoStmt(const clang::GotoStmt *stmt) {
        return bld.compose< hl::GotoStmt >()
            .bind(self.location(stmt))
            .bind_transform(self.visit(stmt->getLabel()), first_result)
            .freeze();
    }

    operation default_stmt_visitor::VisitIndirectGotoStmt(const clang::IndirectGotoStmt *stmt) {
        return bld.compose< hl::IndirectGotoStmt >()
            .bind(self.location(stmt))
            .bind_always(mk_value_builder(stmt->getTarget()))
            .freeze();
    }

    operation default_stmt_visitor::VisitLabelStmt(const clang::LabelStmt *stmt) {
        return bld.compose< hl::LabelStmt >()
            .bind(self.location(stmt))
            .bind_transform(self.visit(stmt->getDecl()), first_result)
            .bind_always(mk_optional_region_builder(stmt->getSubStmt()))
            .freeze();
    }

    operation default_stmt_visitor::VisitIfStmt(const clang::IfStmt *stmt) {
        return bld.compose< hl::IfOp >()
            .bind(self.location(stmt))
            .bind_always(mk_cond_builder(stmt->getCond()))
            .bind_always(mk_optional_region_builder(stmt->getThen()))
            .bind_if(stmt->getElse(), mk_optional_region_builder(stmt->getElse()))
            .freeze();
    }

    //
    // Expressions
    //
    operation default_stmt_visitor::VisitDeclStmt(const clang::DeclStmt *stmt) {
        if (stmt->isSingleDecl()) {
            return self.visit(stmt->getSingleDecl());
        } else {
            operation last = {};
            for (auto decl : stmt->decls()) {
                last = self.visit(decl);
            }
            return last;
        }
    }

    operation default_stmt_visitor::VisitMemberExpr(const clang::MemberExpr *expr) {
        return bld.compose< hl::RecordMemberOp >()
            .bind(self.location(expr))
            .bind(visit_maybe_lvalue_result_type(expr))
            .bind_transform(self.visit(expr->getBase()), first_result)
            .bind(self.symbol(expr->getMemberDecl()))
            .freeze();
    }

    operation default_stmt_visitor::VisitConditionalOperator(const clang::ConditionalOperator *op) {
        return bld.compose< hl::CondOp >()
            .bind(self.location(op))
            .bind(self.visit(op->getType()))
            .bind_always(mk_cond_builder(op->getCond()))
            .bind_always(mk_value_builder(op->getTrueExpr()))
            .bind_always(mk_value_builder(op->getFalseExpr()))
            .freeze();
    }

    operation default_stmt_visitor::VisitChooseExpr(const clang::ChooseExpr *expr) {
        return bld.compose< hl::ChooseExprOp >()
            .bind(self.location(expr))
            // ChooseExpr conserves everything including lvalue-ness
            .bind(visit_maybe_lvalue_result_type(expr))
            .bind_always(mk_cond_builder(expr->getCond()))
            .bind_always(mk_value_builder(expr->getLHS()))
            .bind_always(mk_value_builder(expr->getRHS()))
            .bind_always(expr->isConditionDependent() ? std::nullopt : std::optional(expr->isConditionTrue()))
            .freeze();
    }

    operation default_stmt_visitor::VisitGenericSelectionExpr(const clang::GenericSelectionExpr *expr) {
        auto mk_assoc = [&] (const clang::GenericSelectionExpr::ConstAssociation &assoc) -> operation {
            auto assoc_type = assoc.getType();
            auto assoc_expr = assoc.getAssociationExpr();
            auto type       = assoc_type.isNull() ? mlir::Type() : self.visit(assoc_type);
            return bld.compose< hl::GenericAssocExpr >()
                .bind(self.location(expr))
                .bind(visit_as_maybe_lvalue_type(self, mctx, assoc_expr->getType()))
                .bind(mk_value_builder(assoc_expr))
                .bind_if_valid(type)
                .freeze();
        };
        auto mk_body = [&] (auto &state, auto loc) {
            for (const auto &assoc : expr->associations()) {
                mk_assoc(assoc);
            }
        };

        if (expr->isExprPredicate()) {
            return bld.compose< hl::GenericSelectionExpr >()
                .bind(self.location(expr))
                .bind(visit_maybe_lvalue_result_type(expr))
                .bind(mk_type_yield_builder(expr->getControllingExpr()))
                .bind(mk_body)
                .bind_always(expr->isValueDependent() ? std::nullopt : std::optional(expr->getResultIndex()))
                .freeze();
        }
        if (expr->isTypePredicate()) {
            return bld.compose< hl::GenericSelectionExpr >()
                .bind(self.location(expr))
                .bind(visit_maybe_lvalue_result_type(expr))
                .bind(self.visit(expr->getControllingType()->getType()))
                .bind(mk_body)
                .bind_always(expr->isValueDependent() ? std::nullopt : std::optional(expr->getResultIndex()))
                .freeze();
        }
        VAST_REPORT("Generic expr didn't match any predicate type. Is it valid?");
        expr->dump();
        return {};
    }

    operation default_stmt_visitor::VisitBinaryConditionalOperator(const clang::BinaryConditionalOperator *op) {
        auto common_type = self.visit(op->getCommon()->getType());
        return bld.compose< hl::BinaryCondOp >()
            .bind(self.location(op))
            .bind(self.visit(op->getType()))
            .bind_always(mk_value_builder(op->getCommon()))
            .bind_always(mk_cond_builder_with_args(op->getCond(), common_type))
            .bind_always(mk_value_builder_with_args(op->getTrueExpr(), common_type))
            .bind_always(mk_value_builder(op->getFalseExpr()))
            .freeze();
    }

    operation default_stmt_visitor::VisitAddrLabelExpr(const clang::AddrLabelExpr *expr) {
        return bld.compose< hl::AddrLabelExpr >()
            .bind(self.location(expr))
            .bind(self.visit(expr->getType()))
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
            .bind(visit_as_lvalue_type(self, mctx, expr->getType()))
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
        return bld.constant(self.location(expr).value(), self.visit(expr->getType()), expr->getValue()).getDefiningOp();
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

    operation default_stmt_visitor::mk_direct_call(const clang::CallExpr *expr) {
        if (auto callee = expr->getDirectCallee()) {
            if (callee->getBuiltinID()) {
                auto _ = bld.set_insertion_point_to_start_of_module();
                self.visit(callee);
            }
        }

        return bld.compose< hl::CallOp >()
            .bind(self.location(expr))
            .bind(self.symbol(expr->getDirectCallee()))
            .bind(self.visit(expr->getType()))
            .bind_always(visit_values_range(expr->arguments()))
            .freeze();
    }

    operation default_stmt_visitor::mk_indirect_call(const clang::CallExpr *expr) {
        return bld.compose< hl::IndirectCallOp >()
            .bind(self.location(expr))
            .bind(self.visit(expr->getType()))
            .bind_transform(self.visit(expr->getCallee()), first_result)
            .bind_always(visit_values_range(expr->arguments()))
            .freeze();
    }

    operation default_stmt_visitor::VisitCallExpr(const clang::CallExpr *expr) {
        if (expr->getDirectCallee()) {
            return mk_direct_call(expr);
        } else {
            return mk_indirect_call(expr);
        }
    }

    // operation default_stmt_visitor::VisitCXXMemberCallExpr(const clang::CXXMemberCallExpr *expr)
    // operation default_stmt_visitor::VisitCXXOperatorCallExpr(const clang::CXXOperatorCallExpr *expr)

    operation default_stmt_visitor::VisitOffsetOfExpr(const clang::OffsetOfExpr *expr) {
        attrs_t components;
        std::vector< builder_callback > index_exprs;

        for (unsigned int i = 0; i < expr->getNumComponents(); ++i) {
            auto &component = expr->getComponent(i);
            switch (component.getKind()) {
                case clang::OffsetOfNode::Kind::Array: {
                    auto index = component.getArrayExprIndex();
                    components.push_back(hl::OffsetOfNodeAttr::get(&mctx, index));
                    index_exprs.push_back(mk_value_builder(expr->getIndexExpr(index)));
                    break;
                }
                case clang::OffsetOfNode::Kind::Field: {
                    components.push_back(hl::OffsetOfNodeAttr::get(
                        &mctx,
                        mlir::StringAttr::get(&mctx, self.symbol(component.getField()).value())
                    ));
                    break;
                }
                case clang::OffsetOfNode::Kind::Identifier: {
                    components.push_back(hl::OffsetOfNodeAttr::get(
                        &mctx, mlir::StringAttr::get(&mctx, component.getFieldName()->getName())
                    ));
                    break;
                }
                case clang::OffsetOfNode::Kind::Base: {
                    VAST_REPORT("OffsetOfExprOp unimplemented for C++ classes with inheritance.");
                    return {};
                }
            }
        }

        return bld.compose< hl::OffsetOfExprOp >()
            .bind(self.location(expr))
            .bind(self.visit(expr->getType()))
            .bind(self.visit(expr->getTypeSourceInfo()->getType()))
            .bind(mlir::ArrayAttr::get(&mctx, components))
            .bind_always(index_exprs)
            .freeze();
    }

    operation default_stmt_visitor::VisitOpaqueValueExpr(const clang::OpaqueValueExpr *expr) {
        const auto parents = actx.getParents< clang_stmt >(*expr);
        // In C++ an AST stmt can have multiple parents
        for (const auto &parent : parents) {

            if (mlir::isa< clang::BinaryConditionalOperator >(parent.get< clang_stmt >())) {
                // These arguments should be constructed when creating the BinaryCondOp
                auto args = bld.getBlock()->getParent()->getArguments();
                return bld.compose< hl::OpaqueValueExpr >()
                    .bind(self.location(expr))
                    .bind(self.visit(expr->getType()))
                    .bind(args)
                    .freeze();
            }

        }
        // Right now we will handle the OpaqueValueExpr in case-to-case basis
        // This expression seems to have very "free" semantics and without
        // analysing all the cases it might produce wrong mlir otherwise
        return {};


    }
    // operation default_stmt_visitor::VisitOverloadExpr(const clang::OverloadExpr *expr)

    operation default_stmt_visitor::VisitParenExpr(const clang::ParenExpr *expr) {
        return bld.compose< hl::ExprOp >()
            .bind(self.location(expr))
            .bind_choose(
                expr->isLValue(),
                visit_as_lvalue_type(self, mctx, expr->getType()),
                self.visit(expr->getType())
            )
            .bind_always(mk_value_builder(expr->getSubExpr()))
            .freeze();
    }

    // operation default_stmt_visitor::VisitParenListExpr(const clang::ParenListExpr *expr)

    operation last_effective_operation(mlir::Block *block) {
        auto last = std::prev(block->end());
        while (last != block->begin() && mlir::isa< hl::NullStmt >(&*last)) {
            last = std::prev(last);
        }
        return &*last;
    }
    operation default_stmt_visitor::VisitStmtExpr(const clang::StmtExpr *expr) {
        auto mk_stmt_expr_region_builder = [&] (const clang_stmt *stmt) {
            return [this, stmt] (auto &state, auto) {
                auto compound = clang::cast< clang::CompoundStmt >(stmt);
                for (auto sub_stmt : compound->body()) {
                    self.visit(sub_stmt);
                }

                auto last_block = state.getBlock();
                // ({5;;;;;}) <- this is supposed to return 5...
                auto last = last_effective_operation(last_block);
                auto _ = bld.scoped_insertion_at_end(last_block);

                auto stmt_loc = self.location(stmt).value();

                if (last->getNumResults() > 0) {
                    bld.create< hl::ValueYieldOp >(stmt_loc, last->getResult(0));
                } else {
                    bld.create< hl::ValueYieldOp >(stmt_loc, bld.void_value(stmt_loc));
                }
            };
        };

        return bld.compose< hl::StmtExprOp >()
            .bind(self.location(expr))
            .bind(self.visit(expr->getType()))
            .bind_always(mk_stmt_expr_region_builder(expr->getSubStmt()))
            .freeze();
    }

    operation default_stmt_visitor::VisitUnaryExprOrTypeTraitExpr(const clang::UnaryExprOrTypeTraitExpr *expr) {
        switch (expr->getKind()) {
            case clang::UETT_SizeOf:
                return mk_trait_expr< hl::SizeOfTypeOp, hl::SizeOfExprOp >(expr);
            case clang::UETT_PreferredAlignOf:
                return mk_trait_expr< hl::PreferredAlignOfTypeOp, hl::PreferredAlignOfExprOp >(expr);
            case clang::UETT_AlignOf:
                return mk_trait_expr< hl::AlignOfTypeOp, hl::AlignOfExprOp >(expr);
            default:
                return {};
        }
    }

    operation default_stmt_visitor::VisitTypeTraitExpr(const clang::TypeTraitExpr *expr) {
        switch (expr->getTrait()) {
            case clang::BTT_TypeCompatible:
                return mk_type_trait_expr< hl::BuiltinTypesCompatiblePOp >(expr);
            default:
                return {};
        }
        return {};
    }

    operation default_stmt_visitor::VisitVAArgExpr(const clang::VAArgExpr *expr) {
        return bld.compose< hl::VAArgExpr >()
            .bind(self.location(expr))
            .bind(self.visit(expr->getType()))
            .bind_transform(self.visit(expr->getSubExpr()), results)
            .freeze();
    }

    operation default_stmt_visitor::VisitNullStmt(const clang::NullStmt *stmt) {
        return bld.compose< hl::NullStmt >().bind(self.location(stmt)).freeze();
    }

    operation default_stmt_visitor::VisitCXXThisExpr(const clang::CXXThisExpr *expr) {
        return bld.compose< hl::ThisOp >()
            .bind(self.location(expr))
            .bind(self.visit(expr->getType()))
            .freeze();
    }

    operation default_stmt_visitor::VisitInitListExpr(const clang::InitListExpr *expr) {
        return bld.compose< hl::InitListExpr >()
            .bind(self.location(expr))
            .bind(self.visit(expr->getType()))
            .bind_always(visit_values_range(expr->inits()))
            .freeze();
    }

    operation default_stmt_visitor::VisitImplicitValueInitExpr(const clang::ImplicitValueInitExpr *expr) {
        return bld.compose< hl::InitListExpr >()
            .bind(self.location(expr))
            .bind(self.visit(expr->getType()))
            .bind_always(visit_values_range(expr->children()))
            .freeze();
    }

    operation default_stmt_visitor::VisitAttributedStmt(const clang::AttributedStmt *stmt) {
        auto attr_stmt = bld.compose< hl::AttributedStmt >()
            .bind(self.location(stmt))
            .bind(mk_region_builder(stmt->getSubStmt()))
            .freeze();

        mlir_attr_list attrs;

        // This is not handled by default, because statements usually do not have attributes
        for (auto attr : stmt->getAttrs()) {
            if (auto visited = self.visit(attr)) {
                attrs.set(visited->getName(), visited->getValue());
            }
        }

        attr_stmt->setAttrs(attrs);

        return attr_stmt;
    }

} // namespace vast::cg
