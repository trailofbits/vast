// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <clang/AST/StmtVisitor.h>
#include <clang/AST/OperationKinds.h>
VAST_UNRELAX_WARNINGS

#include "vast/CodeGen/CodeGenMeta.hpp"
#include "vast/CodeGen/CodeGenBuilder.hpp"
#include "vast/CodeGen/CodeGenVisitorBase.hpp"
#include "vast/CodeGen/CodeGenVisitorLens.hpp"

#include "vast/Dialect/HighLevel/HighLevelDialect.hpp"
#include "vast/Dialect/HighLevel/HighLevelOps.hpp"
#include "vast/Dialect/Core/CoreAttributes.hpp"
#include "vast/Dialect/Core/CoreOps.hpp"

#include "vast/Util/Common.hpp"

namespace vast::hl {

    CastKind cast_kind(const clang::CastExpr *expr);
    IdentKind ident_kind(const clang::PredefinedExpr *expr);

} // namespace vast::hl


namespace vast::cg {

    template< typename derived_t >
    struct default_stmt_visitor
        : stmt_visitor_base< default_stmt_visitor< derived_t > >
        , visitor_lens< derived_t, default_stmt_visitor >
    {
        using lens = visitor_lens< derived_t, default_stmt_visitor >;

        using lens::derived;
        using lens::context;
        using lens::mcontext;

        using lens::visit;
        using lens::visit_as_lvalue_type;

        using lens::set_insertion_point_to_start;

        using lens::make_yield_true;
        using lens::make_cond_builder;
        using lens::make_value_builder;
        using lens::make_region_builder;

        using lens::make_value_yield_region;
        using lens::make_stmt_expr_region;

        using lens::insertion_guard;

        using lens::meta_location;

        using lens::constant;

        template< typename Op, typename... Args >
        auto make(Args &&...args) {
            return derived().template create< Op >(std::forward< Args >(args)...);
        }

        operation VisitCompoundStmt(const clang::CompoundStmt *stmt) {
            return derived().template make_scoped< CoreScope >(meta_location(stmt), [&] {
                for (auto s : stmt->body()) {
                    visit(s);
                }
            });
        }

        //
        // Binary Operations
        //

        template< typename Op >
        operation VisitBinOp(const clang::BinaryOperator *op) {
            auto lhs = visit(op->getLHS())->getResult(0);
            auto rhs = visit(op->getRHS())->getResult(0);
            auto type = visit(op->getType());
            return make< Op >(meta_location(op), type, lhs, rhs);
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

        template< typename IOp, typename FOp >
        operation VisitIFBinOp(const clang::BinaryOperator *op) {
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
        operation VisitIFBinOp(const clang::BinaryOperator *op) {
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
        operation VisitCmp(const clang::BinaryOperator *op) {
            auto lhs = visit(op->getLHS())->getResult(0);
            auto rhs = visit(op->getRHS())->getResult(0);
            auto res = visit(op->getType());
            return make< hl::CmpOp >(meta_location(op), res, pred, lhs, rhs);
        }

        template< hl::FPredicate pred >
        operation VisitFCmp(const clang::BinaryOperator *op) {
            auto lhs = visit(op->getLHS())->getResult(0);
            auto rhs = visit(op->getRHS())->getResult(0);
            auto res = visit(op->getType());
            return make< hl::FCmpOp >(meta_location(op), res, pred, lhs, rhs);
        }

        template< hl::Predicate upred, hl::Predicate spred, hl::FPredicate fpred >
        operation VisitCmp(const clang::BinaryOperator *op) {
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

        // operation VisitBinPtrMemD(clang::BinaryOperator *op);
        // operation VisitBinPtrMemI(clang::BinaryOperator *op);

        operation VisitBinMul(const clang::BinaryOperator *op) {
            return VisitIFBinOp< hl::MulIOp, hl::MulFOp >(op);
        }

        operation VisitBinDiv(const clang::BinaryOperator *op) {
            return VisitIFBinOp< hl::DivUOp, hl::DivSOp, hl::DivFOp >(op);
        }

        operation VisitBinRem(const clang::BinaryOperator *op) {
            return VisitIFBinOp< hl::RemUOp, hl::RemSOp, hl::RemFOp >(op);
        }

        operation VisitBinAdd(const clang::BinaryOperator *op) {
            return VisitIFBinOp< hl::AddIOp, hl::AddFOp >(op);
        }

        operation VisitBinSub(const clang::BinaryOperator *op) {
            return VisitIFBinOp< hl::SubIOp, hl::SubFOp >(op);
        }

        operation VisitBinShl(const clang::BinaryOperator *op) {
            return VisitBinOp< hl::BinShlOp >(op);
        }

        operation VisitBinShr(const clang::BinaryOperator *op) {
            return VisitIBinOp< hl::BinLShrOp, hl::BinAShrOp >(op);
        }

        using ipred = hl::Predicate;
        using fpred = hl::FPredicate;

        operation VisitBinLT(const clang::BinaryOperator *op) {
            return VisitCmp< ipred::ult, ipred::slt, fpred::olt >(op);
        }

        operation VisitBinGT(const clang::BinaryOperator *op) {
            return VisitCmp< ipred::ugt, ipred::sgt, fpred::ogt >(op);
        }

        operation VisitBinLE(const clang::BinaryOperator *op) {
            return VisitCmp< ipred::ule, ipred::sle, fpred::ole >(op);
        }

        operation VisitBinGE(const clang::BinaryOperator *op) {
            return VisitCmp< ipred::uge, ipred::sge, fpred::oge >(op);
        }

        operation VisitBinEQ(const clang::BinaryOperator *op) {
            return VisitCmp< ipred::eq, ipred::eq, fpred::oeq >(op);
        }

        operation VisitBinNE(const clang::BinaryOperator *op) {
            return VisitCmp< ipred::ne, ipred::ne, fpred::une >(op);
        }

        operation VisitBinAnd(const clang::BinaryOperator *op) {
            return VisitBinOp< hl::BinAndOp >(op);
        }

        operation VisitBinXor(const clang::BinaryOperator *op) {
            return VisitBinOp< hl::BinXorOp >(op);
        }

        operation VisitBinOr(const clang::BinaryOperator *op) {
            return VisitBinOp< hl::BinOrOp >(op);
        }

        template< typename LOp >
        operation VisitBinLogical(const clang::BinaryOperator *op) {
            auto lhs_builder = make_value_builder(op->getLHS());
            auto rhs_builder = make_value_builder(op->getRHS());
            auto type = visit(op->getType());
            return make< LOp >(meta_location(op), type, lhs_builder, rhs_builder);
        }

        operation VisitBinLAnd(const clang::BinaryOperator *op) {
            return VisitBinLogical< hl::BinLAndOp >(op);
        }

        operation VisitBinLOr(const clang::BinaryOperator *op) {
            return VisitBinLogical< hl::BinLOrOp >(op);
        }

        template< typename Op >
        operation VisitAssignBinOp(const clang::BinaryOperator *op) {
            auto lhs = visit(op->getLHS())->getResult(0);
            auto rhs = visit(op->getRHS())->getResult(0);
            return make< Op >(meta_location(op), lhs, rhs);
        }

        template< typename UOp, typename SOp >
        operation VisitAssignIBinOp(const clang::BinaryOperator *op) {
            auto ty = op->getType();
            if (ty->isUnsignedIntegerType())
                return VisitAssignBinOp< UOp >(op);
            if (ty->isIntegerType())
                return VisitAssignBinOp< SOp >(op);
            return nullptr;
        }

        template< typename IOp, typename FOp >
        operation VisitAssignIFBinOp(const clang::BinaryOperator *op) {
            auto ty = op->getType();
            if (ty->isIntegerType())
                return VisitAssignBinOp< IOp >(op);
            // FIXME: eventually decouple arithmetic and pointer additions?
            if (ty->isPointerType())
                return VisitAssignBinOp< IOp >(op);
            if (ty->isFloatingType())
                return VisitAssignBinOp< FOp >(op);
            return nullptr;
        }

        template< typename UOp, typename SOp, typename FOp >
        operation VisitAssignIFBinOp(const clang::BinaryOperator *op) {
            auto ty = op->getType();
            if (ty->isUnsignedIntegerType())
                return VisitAssignBinOp< UOp >(op);
            if (ty->isIntegerType())
                return VisitAssignBinOp< SOp >(op);
            if (ty->isFloatingType())
                return VisitAssignBinOp< FOp >(op);
            return nullptr;
        }

        operation VisitBinAssign(const clang::BinaryOperator *op) {
            return VisitAssignBinOp< hl::AssignOp >(op);
        }

        //
        // Compound Assignment Operations
        //

        operation VisitBinMulAssign(const clang::CompoundAssignOperator *op) {
            return VisitAssignIFBinOp< hl::MulIAssignOp, hl::MulFAssignOp >(op);
        }

        operation VisitBinDivAssign(const clang::CompoundAssignOperator *op) {
            return VisitAssignIFBinOp< hl::DivUAssignOp, hl::DivSAssignOp, hl::DivFAssignOp >(op);
        }

        operation VisitBinRemAssign(const clang::CompoundAssignOperator *op) {
            return VisitAssignIBinOp< hl::RemUAssignOp, hl::RemSAssignOp >(op);
        }

        operation VisitBinAddAssign(const clang::CompoundAssignOperator *op) {
            return VisitAssignIFBinOp< hl::AddIAssignOp, hl::AddFAssignOp >(op);
        }

        operation VisitBinSubAssign(const clang::CompoundAssignOperator *op) {
            return VisitAssignIFBinOp< hl::SubIAssignOp, hl::SubFAssignOp >(op);
        }

        operation VisitBinShlAssign(const clang::CompoundAssignOperator *op) {
            return VisitAssignBinOp< hl::BinShlAssignOp >(op);
        }

        operation VisitBinShrAssign(const clang::CompoundAssignOperator *op) {
            return VisitAssignIBinOp< hl::BinLShrAssignOp, hl::BinAShrAssignOp >(op);
        }

        operation VisitBinAndAssign(const clang::CompoundAssignOperator *op) {
            return VisitAssignBinOp< hl::BinAndAssignOp >(op);
        }

        operation VisitBinOrAssign(const clang::CompoundAssignOperator *op) {
            return VisitAssignBinOp< hl::BinOrAssignOp >(op);
        }

        operation VisitBinXorAssign(const clang::CompoundAssignOperator *op) {
            return VisitAssignBinOp< hl::BinXorAssignOp >(op);
        }

        operation VisitBinComma(const clang::BinaryOperator *op) {
            auto lhs = visit(op->getLHS())->getResult(0);
            auto rhs = visit(op->getRHS())->getResult(0);
            auto ty  = visit(op->getType());
            return make< hl::BinComma >(meta_location(op), ty, lhs, rhs);
        }

        //
        // Unary Operations
        //

        template< typename Op >
        operation VisitUnary(const clang::UnaryOperator *op, Type rty) {
            auto arg = visit(op->getSubExpr())->getResult(0);
            return make< Op >(meta_location(op), rty, arg);
        }

        template< typename Op >
        operation VisitUnderlyingTypePreservingUnary(const clang::UnaryOperator *op) {
            auto arg = visit(op->getSubExpr())->getResult(0);
            auto type = arg.getType();
            if (auto ltype = type.template dyn_cast< hl::LValueType >()) {
                type = ltype.getElementType();
            }
            return make< Op >(meta_location(op), type, arg);
        }

        operation VisitUnaryPostInc(const clang::UnaryOperator *op) {
            return VisitUnderlyingTypePreservingUnary< hl::PostIncOp >(op);
        }

        operation VisitUnaryPostDec(const clang::UnaryOperator *op) {
            return VisitUnderlyingTypePreservingUnary< hl::PostDecOp >(op);
        }

        operation VisitUnaryPreInc(const clang::UnaryOperator *op) {
            return VisitUnderlyingTypePreservingUnary< hl::PreIncOp >(op);
        }

        operation VisitUnaryPreDec(const clang::UnaryOperator *op) {
            return VisitUnderlyingTypePreservingUnary< hl::PreDecOp >(op);
        }

        operation VisitUnaryAddrOf(const clang::UnaryOperator *op) {
            return VisitUnary< hl::AddressOf >(op, visit(op->getType()));
        }

        operation VisitUnaryDeref(const clang::UnaryOperator *op) {
            return VisitUnary< hl::Deref >(op, visit_as_lvalue_type(op->getType()));
        }

        operation VisitUnaryPlus(const clang::UnaryOperator *op) {
            return VisitUnderlyingTypePreservingUnary< hl::PlusOp >(op);
        }

        operation VisitUnaryMinus(const clang::UnaryOperator *op) {
            return VisitUnderlyingTypePreservingUnary< hl::MinusOp >(op);
        }

        operation VisitUnaryNot(const clang::UnaryOperator *op) {
            return VisitUnderlyingTypePreservingUnary< hl::NotOp >(op);
        }

        operation VisitUnaryLNot(const clang::UnaryOperator *op) {
            return VisitUnary< hl::LNotOp >(op, visit(op->getType()));
        }

        operation VisitUnaryExtension(const clang::UnaryOperator *op) {
            auto visited = visit(op->getSubExpr());
            auto arg = visited->getResult(0);
            return make< hl::ExtensionOp >(meta_location(op), arg.getType(), arg);
        }

        // operation VisitUnaryReal(const clang::UnaryOperator *op)
        // operation VisitUnaryImag(const clang::UnaryOperator *op)
        // operation VisitUnaryCoawait(const clang::UnaryOperator *op)

        //
        // Assembly Statements
        //

        // operation VisitAsmStmt(const clang::AsmStmt *stmt);
        operation VisitGCCAsmStmt(const clang::GCCAsmStmt *stmt) {
            auto get_string_attr = [&](mlir::StringRef str) {
                return mlir::StringAttr::get(&mcontext(), str);
            };

            auto asm_attr = get_string_attr(stmt->getAsmString()->getString());

            if (stmt->isSimple()) {
                return make< hl::AsmOp >(meta_location(stmt),
                                               asm_attr,
                                               stmt->isVolatile(),
                                               false /*has_goto*/
                                              );
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
                return lens::mlir_builder().getI64IntegerAttr(i);
            };

            auto get_out_expr       = [&](int i) { return stmt->getOutputExpr(i); };
            auto get_out_name       = [&](int i) { return stmt->getOutputName(i); };
            auto get_out_constraint = [&](int i) { return stmt->getOutputConstraint(i); };

            auto get_in_expr       = [&](int i) { return stmt->getInputExpr(i); };
            auto get_in_name       = [&](int i) { return stmt->getInputName(i); };
            auto get_in_constraint = [&](int i) { return stmt->getInputConstraint(i); };

            int arg_num = 0;
            auto fill_vectors = [&](int size, const auto &get_expr, const auto &get_name,
                                    const auto &get_constraint, auto &vals, auto &names,
                                    auto &constraints) {
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
                return mlir::ArrayAttr::get(&context().mctx, mlir::ArrayRef(arr));
            };

            return make< hl::AsmOp >(meta_location(stmt),
                                           asm_attr,
                                           stmt->isVolatile(),
                                           stmt->isAsmGoto(),
                                           outputs,
                                           inputs,
                                           get_array_attr(out_names),
                                           get_array_attr(in_names),
                                           get_array_attr(out_constraints),
                                           get_array_attr(in_constraints),
                                           get_array_attr(clobbers),
                                           labels
                                          );
        };
        // operation VisVisitMSAsmStmtitAsmStmt(const clang::MSAsmStmt *stmt)

        //
        // Coroutine Statements
        //

        // operation VisitCoroutineBodyStmt(const clang::CoroutineBodyStmt *stmt)
        // operation VisitCoreturnStmt(const clang::CoreturnStmt *stmt)
        // operation VisitCoroutineSuspendExpr(const clang::CoroutineSuspendExpr *stmt)
        // operation VisitCoawaitExpr(const clang::CoawaitExpr *expr)
        // operation VisitCoyieldExpr(const clang::CoyieldExpr *expr)
        // operation VisitDependentCoawaitExpr(const clang::DependentCoawaitExpr *expr)

        // operation VisitAttributedStmt(const clang::AttributedStmt *stmt)

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
                default: return unsupported_cast();
            }
        }


        template< typename Cast >
        operation VisitCast(const clang::CastExpr *expr) {
            auto arg = visit(expr->getSubExpr());
            auto rty = VisitCastReturnType(expr, arg->getResultTypes().front());
            return make< Cast >(meta_location(expr), rty, arg->getResult(0), hl::cast_kind(expr));
        }

        operation VisitImplicitCastExpr(const clang::ImplicitCastExpr *expr) {
             return VisitCast< hl::ImplicitCastOp >(expr);
        }

        operation VisitCStyleCastExpr(const clang::CStyleCastExpr *expr) {
            return VisitCast< hl::CStyleCastOp >(expr);
        }

        operation VisitBuiltinBitCastExpr(const clang::BuiltinBitCastExpr *expr) {
            return VisitCast< hl::BuiltinBitCastOp >(expr);
        }

        // operation VisitCXXFunctionalCastExpr(const clang::CXXFunctionalCastExpr *expr)
        // operation VisitCXXConstCastExpr(const clang::CXXConstCastExpr *expr)
        // operation VisitCXXDynamicCastExpr(const clang::CXXDynamicCastExpr *expr)
        // operation VisitCXXReinterpretCastExpr(const clang::CXXReinterpretCastExpr *expr)
        // operation VisitCXXStaticCastExpr(const clang::CXXStaticCastExpr *expr)

        //
        // Other Expressions
        //

        operation VisitDeclStmt(const clang::DeclStmt *stmt) {
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

        hl::VarDeclOp getDefiningOpOfGlobalVar(const clang::VarDecl *decl) {
            return context().vars.lookup(decl).template getDefiningOp< hl::VarDeclOp >();
        }

        operation VisitEnumDeclRefExpr(const clang::DeclRefExpr *expr, const clang::Decl *underlying_decl) {
            auto decl = clang::cast< clang::EnumConstantDecl >( underlying_decl )->getFirstDecl();
            if (auto val = context().enumconsts.lookup(decl)) {
                auto rty = visit(expr->getType());
                return make< hl::EnumRefOp >(meta_location(expr), rty, val.getName());
            }

            // Ref: https://github.com/trailofbits/vast/issues/384
            // github issue to avoid emitting error if declaration is missing
            context().error("error: missing enum constant declaration " + decl->getName());
            return nullptr;
        }

        operation VisitVarDeclRefExprImpl(const clang::DeclRefExpr *expr, Value var) {
            auto rty = getLValueReturnType(expr);
            return make< hl::DeclRefOp >(meta_location(expr), rty, var);
        }

        operation VisitVarDeclRefExpr(const clang::DeclRefExpr *expr, const clang::Decl *underlying_decl) {
            auto decl = clang::cast< clang::VarDecl >( underlying_decl )->getFirstDecl();
            if (auto var = context().vars.lookup(decl)) {
                return VisitVarDeclRefExprImpl(expr, var);
            }

            // Ref: https://github.com/trailofbits/vast/issues/384
            // github issue to avoid emitting error if declaration is missing
            context().error("error: missing variable declaration " + decl->getName());
            return nullptr;
        }

        operation VisitFileVarDeclRefExpr(const clang::DeclRefExpr *expr, const clang::Decl *underlying_decl) {
            auto decl = clang::cast< clang::VarDecl >( underlying_decl )->getFirstDecl();
            if (!context().vars.lookup(decl)) {
                // Ref: https://github.com/trailofbits/vast/issues/384
                // github issue to avoid emitting error if declaration is missing
                context().error("error: missing global variable declaration " + decl->getName());
                return nullptr;
            }
            auto var  = getDefiningOpOfGlobalVar(decl);
            auto name = mlir::StringAttr::get(&mcontext(), var.getName());

            auto rty = getLValueReturnType(expr);
            return make< hl::GlobalRefOp >(meta_location(expr), rty, name);
        }

        operation VisitFunctionDeclRefExpr(const clang::DeclRefExpr *expr, const clang::Decl *underlying_decl) {
            auto decl = clang::cast< clang::FunctionDecl >( underlying_decl )->getFirstDecl();
            auto mangled = context().get_mangled_name(decl);
            auto fn      = context().lookup_function(mangled, false);
            if (!fn) {
                auto guard = insertion_guard();
                set_insertion_point_to_start(&context().getBodyRegion());
                fn = mlir::cast< hl::FuncOp >(visit(decl));
            }
            auto rty = getLValueReturnType(expr);

            return make< hl::FuncRefOp >(meta_location(expr), rty, mlir::SymbolRefAttr::get(fn));
        }

        operation VisitDeclRefExpr(const clang::DeclRefExpr *expr) {
            auto underlying = expr->getDecl()->getUnderlyingDecl();

            if (clang::isa< clang::EnumConstantDecl >(underlying)) {
                return VisitEnumDeclRefExpr(expr, underlying);
            }

            if (auto decl = clang::dyn_cast< clang::VarDecl >(underlying)) {
                if (decl->isFileVarDecl())
                    return VisitFileVarDeclRefExpr(expr, underlying);
                return VisitVarDeclRefExpr(expr, underlying);
            }

            if (clang::isa< clang::FunctionDecl >(underlying)) {
                return VisitFunctionDeclRefExpr(expr, underlying);
            }

            VAST_UNIMPLEMENTED_MSG("unknown underlying declaration to be referenced");
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

        operation VisitReturnStmt(const clang::ReturnStmt *stmt) {
            auto loc = meta_location(stmt);
            if (auto ret = stmt->getRetValue())
                return make< hl::ReturnOp >(loc, visit(ret)->getResults());
            auto void_val = constant(loc);
            return make< hl::ReturnOp >(loc, void_val);
        }


        operation VisitBreakStmt(const clang::BreakStmt *stmt) {
            return make< hl::BreakOp >(meta_location(stmt));
        }

        operation VisitContinueStmt(const clang::ContinueStmt *stmt) {
            return make< hl::ContinueOp >(meta_location(stmt));
        }

        operation VisitCaseStmt(const clang::CaseStmt *stmt) {
            auto lhs_builder  = make_value_builder(stmt->getLHS());
            auto body_builder = make_region_builder(stmt->getSubStmt());
            return make< hl::CaseOp >(meta_location(stmt), lhs_builder, body_builder);
        }

        operation VisitDefaultStmt(const clang::DefaultStmt *stmt) {
            auto body_builder = make_region_builder(stmt->getSubStmt());
            return make< hl::DefaultOp >(meta_location(stmt), body_builder);
        }

        operation VisitSwitchStmt(const clang::SwitchStmt *stmt) {
            auto loc = meta_location(stmt);

            auto make_switch_op = [&] {
                auto cond_builder = make_value_builder(stmt->getCond());
                auto body_builder = make_region_builder(stmt->getBody());
                return make< hl::SwitchOp >(loc, cond_builder, body_builder);
            };

            if (stmt->getInit()) {
                return derived().template make_scoped< CoreScope >(loc, [&] {
                    visit(stmt->getInit());
                    make_switch_op();
                });
            }

            return make_switch_op();
        }

        operation VisitDoStmt(const clang::DoStmt *stmt) {
            auto cond_builder = make_cond_builder(stmt->getCond());
            auto body_builder = make_region_builder(stmt->getBody());
            return make< hl::DoOp >(meta_location(stmt), body_builder, cond_builder);
        }

        operation VisitWhileStmt(const clang::WhileStmt *stmt) {
            auto cond_builder = make_cond_builder(stmt->getCond());
            auto body_builder = make_region_builder(stmt->getBody());
            return make< hl::WhileOp >(meta_location(stmt), cond_builder, body_builder);
        }

        // operation VisitCXXCatchStmt(const clang::CXXCatchStmt *stmt)
        // operation VisitCXXForRangeStmt(const clang::CXXForRangeStmt *stmt)
        // operation VisitCXXTryStmt(const clang::CXXTryStmt *stmt)
        // operation VisitCXXTryStmt(const clang::CXXTryStmt *stmt)
        // operation VisitCapturedStmt(const clang::CapturedStmt *stmt)

        operation VisitForStmt(const clang::ForStmt *stmt) {
            auto loc = meta_location(stmt);

            auto make_loop_op = [&] {
                auto incr = make_region_builder(stmt->getInc());
                auto body = make_region_builder(stmt->getBody());
                if (auto cond = stmt->getCond())
                    return make< hl::ForOp >(loc, make_cond_builder(cond), incr, body);
                return make< hl::ForOp >(loc, make_yield_true(), incr, body);
            };

            if (stmt->getInit()) {
                return derived().template make_scoped< CoreScope >(loc, [&] {
                    visit(stmt->getInit());
                    make_loop_op();
                });
            }

            return make_loop_op();
        }

        operation VisitGotoStmt(const clang::GotoStmt *stmt) {
            auto lab = visit(stmt->getLabel())->getResult(0);
            return make< hl::GotoStmt >(meta_location(stmt), lab);
        }
        // operation VisitIndirectGotoStmt(const clang::IndirectGotoStmt *stmt)

        operation VisitLabelStmt(const clang::LabelStmt *stmt) {
            auto lab = visit(stmt->getDecl())->getResult(0);
            auto sub_builder = make_region_builder(stmt->getSubStmt());
            return make< hl::LabelStmt >(meta_location(stmt), lab, sub_builder);
        }

        operation VisitIfStmt(const clang::IfStmt *stmt) {
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

        operation VisitMemberExpr(const clang::MemberExpr *expr) {
            auto name = context().get_decl_name(expr->getMemberDecl());
            auto base = visit(expr->getBase())->getResult(0);
            auto type = visit_as_lvalue_type(expr->getType());
            return make< hl::RecordMemberOp >(meta_location(expr), type, base, name);
        }

        // operation VisitAbstractConditionalOperator(const clang::AbstractConditionalOperator *op)
        // operation VisitAbstractConditionalOperator(const clang::BinaryConditionalOperator *op)

        operation VisitConditionalOperator(const clang::ConditionalOperator *op) {
            auto type = visit(op->getType());
            auto cond = make_cond_builder(op->getCond());
            auto true_expr = make_value_builder(op->getTrueExpr());
            auto false_expr = make_value_builder(op->getFalseExpr());
            return make< hl::CondOp >(meta_location(op), type, cond, true_expr, false_expr);
        }

        operation VisitAddrLabelExpr(const clang::AddrLabelExpr *expr) {
            auto lab = visit(expr->getLabel())->getResult(0);
            auto rty = visit_as_lvalue_type(expr->getType());
            return make< hl::AddrLabelExpr >(meta_location(expr), rty, lab);
        }

        operation VisitConstantExpr(const clang::ConstantExpr *expr) {
            // TODO(Heno): crete hl.constantexpr
            return visit(expr->getSubExpr());
        }

        operation VisitArraySubscriptExpr(const clang::ArraySubscriptExpr *expr) {
            auto rty    = visit_as_lvalue_type(expr->getType());
            auto base   = visit(expr->getBase())->getResult(0);
            auto offset = visit(expr->getIdx())->getResult(0);
            return make< hl::SubscriptOp >(meta_location(expr), rty, base, offset);
        }

        // operation VisitArrayTypeTraitExpr(const clang::ArrayTypeTraitExpr *expr)
        // operation VisitAsTypeExpr(const clang::AsTypeExpr *expr)
        // operation VisitAtomicExpr(const clang::AtomicExpr *expr)
        // operation VisitBlockExpr(const clang::BlockExpr *expr)

        // operation VisitCXXBindTemporaryExpr(const clang::CXXBindTemporaryExpr *expr)

        operation VisitCXXBoolLiteralExpr(const clang::CXXBoolLiteralExpr *lit) {
            return VisitScalarLiteral(lit, lit->getValue());
        }

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

        hl::FuncOp VisitDirectCallee(const clang::FunctionDecl *callee) {
            auto guard = insertion_guard();

            auto mangled = context().get_mangled_name(callee);
            if (auto fn = context().lookup_function(mangled, false /* with error */)) {
                return fn;
            }

            set_insertion_point_to_start(&context().getBodyRegion());
            return mlir::cast< hl::FuncOp >(visit(callee));
        }

        operation VisitIndirectCallee(const clang::Expr *callee) {
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

        operation VisitDirectCall(const clang::CallExpr *expr, const clang::Decl *decl) {
            auto callee = VisitDirectCallee(clang::cast< clang::FunctionDecl >( decl ));
            auto args   = VisitArguments(expr);
            return make< hl::CallOp >(meta_location(expr), callee, args);
        }

        operation VisitIndirectCall(const clang::CallExpr *expr) {
            auto callee = VisitIndirectCallee(expr->getCallee())->getResult(0);
            auto args   = VisitArguments(expr);
            auto type   = hl::getFunctionType(callee.getType(), context().mod.get());
            if (type) {
                return make< hl::IndirectCallOp >(
                    meta_location(expr), type.getResults(), callee, args
                );
            }

            return {};
        }

        operation VisitCallExpr(const clang::CallExpr *expr) {
            if (auto callee = expr->getDirectCallee()) {
                return VisitDirectCall(expr, callee->getFirstDecl());
            }

            return VisitIndirectCall(expr);
        }

        // operation VisitCXXMemberCallExpr(const clang::CXXMemberCallExpr *expr)
        // operation VisitCXXOperatorCallExpr(const clang::CXXOperatorCallExpr *expr)

        // operation VisitOffsetOfExpr(const clang::OffsetOfExpr *expr)
        // operation VisitOpaqueValueExpr(const clang::OpaqueValueExpr *expr)
        // operation VisitOverloadExpr(const clang::OverloadExpr *expr)

        operation VisitParenExpr(const clang::ParenExpr *expr) {
            auto [reg, rty] = make_value_yield_region(expr->getSubExpr());
            return make< hl::ExprOp >(meta_location(expr), rty, std::move(reg));
        }

        // operation VisitParenListExpr(const clang::ParenListExpr *expr)
        operation VisitStmtExpr(const clang::StmtExpr *expr) {
            auto loc = meta_location(expr);
            auto sub = llvm::cast< clang::CompoundStmt >(expr->getSubStmt());
            auto [reg, rty] = make_stmt_expr_region(sub);
            return make< hl::StmtExprOp >(loc, rty, std::move(reg));
        }

        template< typename Op >
        operation ExprTypeTrait(const clang::UnaryExprOrTypeTraitExpr *expr, auto rty, auto loc) {
            auto arg = make_value_builder(expr->getArgumentExpr());
            return make< Op >(loc, rty, arg);
        }

        template< typename Op >
        operation TypeTraitExpr(const clang::UnaryExprOrTypeTraitExpr *expr, auto rty, auto loc) {
            auto arg = visit(expr->getArgumentType());
            return make< Op >(loc, rty, arg);
        }

        template< typename TypeTraitOp, typename ExprTraitOp >
        operation VisitTraitExpr(const clang::UnaryExprOrTypeTraitExpr *expr) {
            auto loc = meta_location(expr);
            auto rty = visit(expr->getType());

            return expr->isArgumentType() ? TypeTraitExpr< TypeTraitOp >(expr, rty, loc)
                                          : ExprTypeTrait< ExprTraitOp >(expr, rty, loc);
        }

        operation VisitUnaryExprOrTypeTraitExpr(const clang::UnaryExprOrTypeTraitExpr *expr) {
            auto kind = expr->getKind();

            if (kind == clang::UETT_SizeOf) {
                return VisitTraitExpr< hl::SizeOfTypeOp, hl::SizeOfExprOp >(expr);
            }

            if (kind == clang::UETT_AlignOf) {
                return VisitTraitExpr< hl::AlignOfTypeOp, hl::AlignOfExprOp >(expr);
            }

            return {};
        }

        operation VisitVAArgExpr(const clang::VAArgExpr *expr) {
            auto loc = meta_location(expr);
            auto rty = visit(expr->getType());
            auto arg = visit(expr->getSubExpr())->getResults();
            return make< hl::VAArgExpr >(loc, rty, arg);
        }

        operation VisitNullStmt(const clang::NullStmt *stmt) {
            return make< hl::SkipStmt >(meta_location(stmt));
        }

        operation VisitCXXThisExpr(const clang::CXXThisExpr *expr) {
            auto rty = visit(expr->getType());
            return make< hl::ThisOp >(meta_location(expr), rty);
        }

        //
        // Literals
        //

        template< typename LiteralType, typename Value >
        operation VisitScalarLiteral(const LiteralType *lit, Value value) {
            if constexpr (std::is_same_v< Value, bool >) {
                return constant(meta_location(lit), value).getDefiningOp();
            } else {
                // in C string literals are arrays and therefore lvalues
                auto type = lit->isLValue() ? visit_as_lvalue_type(lit->getType())
                                            : visit(lit->getType());
                return constant(meta_location(lit), type, value).getDefiningOp();
            }
        }

        operation VisitCharacterLiteral(const clang::CharacterLiteral *lit) {
            return VisitScalarLiteral(lit, apsint(lit->getValue()));
        }

        operation VisitIntegerLiteral(const clang::IntegerLiteral *lit) {
            return VisitScalarLiteral(lit, llvm::APSInt(lit->getValue(), false));
        }

        operation VisitFloatingLiteral(const clang::FloatingLiteral *lit) {
            return VisitScalarLiteral(lit, lit->getValue());
        }

        operation VisitStringLiteral(const clang::StringLiteral *lit) {
            return VisitScalarLiteral(lit, lit->getString());
        }

        // operation VisitUserDefinedLiteral(const clang::UserDefinedLiteral *lit)
        // operation VisitCompoundLiteralExpr(const clang::CompoundLiteralExpr *lit)
        // operation VisitFixedPointLiteral(const clang::FixedPointLiteral *lit)

        operation VisitInitListExpr(const clang::InitListExpr *expr) {
            auto ty = visit(expr->getType());

            llvm::SmallVector< Value > elements;
            for (auto elem : expr->inits()) {
                elements.push_back(visit(elem)->getResult(0));
            }

            return make< hl::InitListExpr >(meta_location(expr), ty, elements);
        }
    };

} // namespace vast::cg
