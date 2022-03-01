// Copyright (c) 2022-present, Trail of Bits, Inc.

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
VAST_UNRELAX_WARNINGS

#include "vast/Translation/HighLevelVisitor.hpp"

namespace vast::hl
{
    // Binary Operations

    ValueOrStmt CodeGenVisitor::VisitBinPtrMemD(clang::BinaryOperator *expr) {
        UNREACHABLE("unsupported BinPtrMemD");
    }

    ValueOrStmt CodeGenVisitor::VisitBinPtrMemI(clang::BinaryOperator *expr) {
        UNREACHABLE("unsupported BinPtrMemI");
    }

    ValueOrStmt CodeGenVisitor::VisitBinMul(clang::BinaryOperator *expr) {
        return checked(make_ibin< MulIOp >(expr));
    }

    ValueOrStmt CodeGenVisitor::VisitBinDiv(clang::BinaryOperator *expr) {
        return checked(make_ibin< DivUOp, DivSOp >(expr));
    }

    ValueOrStmt CodeGenVisitor::VisitBinRem(clang::BinaryOperator *expr) {
        return checked(make_ibin< RemUOp, RemSOp >(expr));
    }

    ValueOrStmt CodeGenVisitor::VisitBinAdd(clang::BinaryOperator *expr) {
        return checked(make_ibin< AddIOp >(expr));
    }

    ValueOrStmt CodeGenVisitor::VisitBinSub(clang::BinaryOperator *expr) {
        return checked(make_ibin< SubIOp >(expr));
    }

    ValueOrStmt CodeGenVisitor::VisitBinShl(clang::BinaryOperator *expr) {
        return checked(make_ibin< BinShlOp >(expr));
    }

    ValueOrStmt CodeGenVisitor::VisitBinShr(clang::BinaryOperator *expr) {
        return checked(make_ibin< BinShrOp >(expr));
    }

    ValueOrStmt CodeGenVisitor::VisitBinLT(clang::BinaryOperator *expr) {
        return checked(make_icmp< Predicate::ult, Predicate::slt >(expr));
    }

    ValueOrStmt CodeGenVisitor::VisitBinGT(clang::BinaryOperator *expr) {
        return checked(make_icmp< Predicate::ugt, Predicate::sgt >(expr));
    }

    ValueOrStmt CodeGenVisitor::VisitBinLE(clang::BinaryOperator *expr) {
        return checked(make_icmp< Predicate::ule, Predicate::sle >(expr));
    }

    ValueOrStmt CodeGenVisitor::VisitBinGE(clang::BinaryOperator *expr) {
        return checked(make_icmp< Predicate::uge, Predicate::sge >(expr));
    }

    ValueOrStmt CodeGenVisitor::VisitBinEQ(clang::BinaryOperator *expr) {
        return checked(make_icmp< Predicate::eq >(expr));
    }

    ValueOrStmt CodeGenVisitor::VisitBinNE(clang::BinaryOperator *expr) {
        return checked(make_icmp< Predicate::ne >(expr));
    }

    ValueOrStmt CodeGenVisitor::VisitBinAnd(clang::BinaryOperator *expr) {
        return checked(make_ibin< BinAndOp >(expr));
    }

    ValueOrStmt CodeGenVisitor::VisitBinXor(clang::BinaryOperator *expr) {
        return checked(make_ibin< BinXorOp >(expr));
    }

    ValueOrStmt CodeGenVisitor::VisitBinOr(clang::BinaryOperator *expr) {
        return checked(make_ibin< BinOrOp >(expr));
    }

    ValueOrStmt CodeGenVisitor::VisitBinLAnd(clang::BinaryOperator *expr) {
        return checked(make_ibin< BinLAndOp >(expr));
    }

    ValueOrStmt CodeGenVisitor::VisitBinLOr(clang::BinaryOperator *expr) {
        return checked(make_ibin< BinLOrOp >(expr));
    }

    ValueOrStmt CodeGenVisitor::VisitBinAssign(clang::BinaryOperator *expr) {
        return make_bin< AssignOp >(expr);
    }

    // Compound Assignment Operations

    ValueOrStmt CodeGenVisitor::VisitBinMulAssign(clang::CompoundAssignOperator *expr) {
        return checked(make_ibin< MulIAssignOp >(expr));
    }

    ValueOrStmt CodeGenVisitor::VisitBinDivAssign(clang::CompoundAssignOperator *expr) {
        return checked(make_ibin< DivUAssignOp, DivSAssignOp >(expr));
    }

    ValueOrStmt CodeGenVisitor::VisitBinRemAssign(clang::CompoundAssignOperator *expr) {
        return checked(make_ibin< RemUAssignOp, RemSAssignOp >(expr));
    }

    ValueOrStmt CodeGenVisitor::VisitBinAddAssign(clang::CompoundAssignOperator *expr) {
        return checked(make_ibin< AddIAssignOp >(expr));
    }

    ValueOrStmt CodeGenVisitor::VisitBinSubAssign(clang::CompoundAssignOperator *expr) {
        return checked(make_ibin< SubIAssignOp >(expr));
    }

    ValueOrStmt CodeGenVisitor::VisitBinShlAssign(clang::CompoundAssignOperator *expr) {
        return checked(make_ibin< BinShlAssignOp >(expr));
    }

    ValueOrStmt CodeGenVisitor::VisitBinShrAssign(clang::CompoundAssignOperator *expr) {
        return checked(make_ibin< BinShrAssignOp >(expr));
    }

    ValueOrStmt CodeGenVisitor::VisitBinAndAssign(clang::CompoundAssignOperator *expr) {
        return checked(make_ibin< BinAndAssignOp >(expr));
    }

    ValueOrStmt CodeGenVisitor::VisitBinOrAssign(clang::CompoundAssignOperator *expr) {
        return checked(make_ibin< BinOrAssignOp >(expr));
    }

    ValueOrStmt CodeGenVisitor::VisitBinXorAssign(clang::CompoundAssignOperator *expr) {
        return checked(make_ibin< BinXorAssignOp >(expr));
    }

    ValueOrStmt CodeGenVisitor::VisitBinComma(clang::BinaryOperator *expr) {
        UNREACHABLE("unsupported BinComma");
    }

    // Unary Operations

    ValueOrStmt CodeGenVisitor::VisitUnaryPostInc(clang::UnaryOperator *expr) {
        return make_type_preserving_unary< PostIncOp >(expr);
    }

    ValueOrStmt CodeGenVisitor::VisitUnaryPostDec(clang::UnaryOperator *expr) {
        return make_type_preserving_unary< PostDecOp >(expr);
    }

    ValueOrStmt CodeGenVisitor::VisitUnaryPreInc(clang::UnaryOperator *expr) {
        return make_type_preserving_unary< PreIncOp >(expr);
    }

    ValueOrStmt CodeGenVisitor::VisitUnaryPreDec(clang::UnaryOperator *expr) {
        return make_type_preserving_unary< PreDecOp >(expr);
    }

    ValueOrStmt CodeGenVisitor::VisitUnaryAddrOf(clang::UnaryOperator *expr) {
        return make_unary< AddressOf >(expr);
    }

    ValueOrStmt CodeGenVisitor::VisitUnaryDeref(clang::UnaryOperator *expr) {
        return make_unary< Deref >(expr);
    }

    ValueOrStmt CodeGenVisitor::VisitUnaryPlus(clang::UnaryOperator *expr) {
        return make_type_preserving_unary< PlusOp >(expr);
    }

    ValueOrStmt CodeGenVisitor::VisitUnaryMinus(clang::UnaryOperator *expr) {
        return make_type_preserving_unary< MinusOp >(expr);
    }

    ValueOrStmt CodeGenVisitor::VisitUnaryNot(clang::UnaryOperator *expr) {
        return make_type_preserving_unary< NotOp >(expr);
    }

    ValueOrStmt CodeGenVisitor::VisitUnaryLNot(clang::UnaryOperator *expr) {
        return make_type_preserving_unary< LNotOp >(expr);
    }

    ValueOrStmt CodeGenVisitor::VisitUnaryReal(clang::UnaryOperator *expr) {
        UNREACHABLE("unsupported UnaryReal");
    }

    ValueOrStmt CodeGenVisitor::VisitUnaryImag(clang::UnaryOperator *expr) {
        UNREACHABLE("unsupported UnaryImag");
    }

    ValueOrStmt CodeGenVisitor::VisitUnaryExtension(clang::UnaryOperator *expr) {
        UNREACHABLE("unsupported UnaryExtension");
    }

    ValueOrStmt CodeGenVisitor::VisitUnaryCoawait(clang::UnaryOperator *expr) {
        UNREACHABLE("unsupported UnaryCoawait");
    }

    // Assembly Statements

    ValueOrStmt CodeGenVisitor::VisitAsmStmt(clang::AsmStmt *stmt) {
        UNREACHABLE("unsupported AsmStmt");
    }

    ValueOrStmt CodeGenVisitor::VisitGCCAsmStmt(clang::GCCAsmStmt *stmt) {
        UNREACHABLE("unsupported GCCAsmStmt");
    }

    ValueOrStmt CodeGenVisitor::VisitMSAsmStmt(clang::MSAsmStmt *stmt) {
        UNREACHABLE("unsupported MSAsmStmt");
    }

    ValueOrStmt CodeGenVisitor::VisitCoroutineBodyStmt(clang::CoroutineBodyStmt *stmt) {
        UNREACHABLE("unsupported CoroutineBodyStmt");
    }

    ValueOrStmt CodeGenVisitor::VisitCoreturnStmt(clang::CoreturnStmt *stmt) {
        UNREACHABLE("unsupported CoreturnStmt");
    }

    ValueOrStmt CodeGenVisitor::VisitCoroutineSuspendExpr(clang::CoroutineSuspendExpr *expr) {
        UNREACHABLE("unsupported CoroutineSuspendExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitCoawaitExpr(clang::CoawaitExpr *expr) {
        UNREACHABLE("unsupported CoawaitExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitCoyieldExpr(clang::CoyieldExpr *expr) {
        UNREACHABLE("unsupported CoyieldExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitDependentCoawaitExpr(clang::DependentCoawaitExpr *expr) {
        UNREACHABLE("unsupported DependentCoawaitExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitAttributedStmt(clang::AttributedStmt *stmt) {
        UNREACHABLE("unsupported AttributedStmt");
    }

    // Statements

    ValueOrStmt CodeGenVisitor::VisitBreakStmt(clang::BreakStmt *stmt) {
        auto loc = builder.get_location(stmt->getSourceRange());
        return builder.make< BreakOp >(loc);
    }

    ValueOrStmt CodeGenVisitor::VisitCXXCatchStmt(clang::CXXCatchStmt *stmt) {
        UNREACHABLE("unsupported CXXCatchStmt");
    }

    ValueOrStmt CodeGenVisitor::VisitCXXForRangeStmt(clang::CXXForRangeStmt *stmt) {
        UNREACHABLE("unsupported CXXForRangeStmt");
    }

    ValueOrStmt CodeGenVisitor::VisitCXXTryStmt(clang::CXXTryStmt *stmt) {
        UNREACHABLE("unsupported CXXTryStmt");
    }

    ValueOrStmt CodeGenVisitor::VisitCapturedStmt(clang::CapturedStmt *stmt) {
        UNREACHABLE("unsupported CapturedStmt");
    }

    ValueOrStmt CodeGenVisitor::VisitCompoundStmt(clang::CompoundStmt *stmt) {
        ScopedInsertPoint builder_scope(builder);

        auto loc = builder.get_location(stmt->getSourceRange());

        ScopeOp scope = builder.make< ScopeOp >(loc);
        auto &body    = scope.body();
        body.push_back(new mlir::Block());
        builder.set_insertion_point_to_start(&body.front());

        for (auto s : stmt->body()) {
            CodeGenVisitor::Visit(s);
        }

        return scope;
    }

    ValueOrStmt CodeGenVisitor::VisitContinueStmt(clang::ContinueStmt *stmt) {
        auto loc = builder.get_location(stmt->getSourceRange());
        return builder.make< ContinueOp >(loc);
    }

    ValueOrStmt CodeGenVisitor::VisitDeclStmt(clang::DeclStmt *stmt) {
        ValueOrStmt last;
        for (auto decl : stmt->decls()) {
            last = CodeGenVisitor::Visit(decl);
        }
        return last;
    }

    ValueOrStmt CodeGenVisitor::VisitDoStmt(clang::DoStmt *stmt) {
        auto loc          = builder.get_location(stmt->getSourceRange());
        auto cond_builder = make_cond_builder(stmt->getCond());
        auto body_builder = make_region_builder(stmt->getBody());
        return builder.make< DoOp >(loc, body_builder, cond_builder);
    }

    // Expressions

    ValueOrStmt CodeGenVisitor::VisitAbstractConditionalOperator(
        clang::AbstractConditionalOperator *stmt) {
        UNREACHABLE("unsupported AbstractConditionalOperator");
    }

    ValueOrStmt CodeGenVisitor::VisitBinaryConditionalOperator(
        clang::BinaryConditionalOperator *stmt) {
        UNREACHABLE("unsupported BinaryConditionalOperator");
    }

    ValueOrStmt CodeGenVisitor::VisitConditionalOperator(clang::ConditionalOperator *stmt) {
        UNREACHABLE("unsupported ConditionalOperator");
    }

    ValueOrStmt CodeGenVisitor::VisitAddrLabelExpr(clang::AddrLabelExpr *expr) {
        UNREACHABLE("unsupported AddrLabelExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitConstantExpr(clang::ConstantExpr *expr) {
        return CodeGenVisitor::Visit(expr->getSubExpr());
    }

    ValueOrStmt CodeGenVisitor::VisitArraySubscriptExpr(clang::ArraySubscriptExpr *expr) {
        auto loc    = builder.get_location(expr->getSourceRange());
        auto rty    = types.convert(expr->getType());
        auto base   = CodeGenVisitor::Visit(expr->getBase());
        auto offset = CodeGenVisitor::Visit(expr->getIdx());
        return builder.make_value< SubscriptOp >(loc, rty, base, offset);
    }

    ValueOrStmt CodeGenVisitor::VisitArrayTypeTraitExpr(clang::ArrayTypeTraitExpr *expr) {
        UNREACHABLE("unsupported ArrayTypeTraitExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitAsTypeExpr(clang::AsTypeExpr *expr) {
        UNREACHABLE("unsupported AsTypeExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitAtomicExpr(clang::AtomicExpr *expr) {
        UNREACHABLE("unsupported AtomicExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitBlockExpr(clang::BlockExpr *expr) {
        UNREACHABLE("unsupported BlockExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitCXXBindTemporaryExpr(clang::CXXBindTemporaryExpr *expr) {
        UNREACHABLE("unsupported CXXBindTemporaryExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitCXXBoolLiteralExpr(const clang::CXXBoolLiteralExpr *lit) {
        return make_scalar_literal(lit);
    }

    ValueOrStmt CodeGenVisitor::VisitCXXConstructExpr(clang::CXXConstructExpr *expr) {
        UNREACHABLE("unsupported CXXConstructExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitCXXTemporaryObjectExpr(clang::CXXTemporaryObjectExpr *expr) {
        UNREACHABLE("unsupported CXXTemporaryObjectExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitCXXDefaultArgExpr(clang::CXXDefaultArgExpr *expr) {
        UNREACHABLE("unsupported CXXDefaultArgExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitCXXDefaultInitExpr(clang::CXXDefaultInitExpr *expr) {
        UNREACHABLE("unsupported CXXDefaultInitExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitCXXDeleteExpr(clang::CXXDeleteExpr *expr) {
        UNREACHABLE("unsupported CXXDeleteExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitCXXDependentScopeMemberExpr(
        clang::CXXDependentScopeMemberExpr *expr) {
        UNREACHABLE("unsupported CXXDependentScopeMemberExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitCXXNewExpr(clang::CXXNewExpr *expr) {
        UNREACHABLE("unsupported CXXNewExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitCXXNoexceptExpr(clang::CXXNoexceptExpr *expr) {
        UNREACHABLE("unsupported CXXNoexceptExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitCXXNullPtrLiteralExpr(clang::CXXNullPtrLiteralExpr *expr) {
        UNREACHABLE("unsupported CXXNullPtrLiteralExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitCXXPseudoDestructorExpr(clang::CXXPseudoDestructorExpr *expr) {
        UNREACHABLE("unsupported CXXPseudoDestructorExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitCXXScalarValueInitExpr(clang::CXXScalarValueInitExpr *expr) {
        UNREACHABLE("unsupported CXXScalarValueInitExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitCXXStdInitializerListExpr(
        clang::CXXStdInitializerListExpr *expr) {
        UNREACHABLE("unsupported CXXStdInitializerListExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitCXXThisExpr(clang::CXXThisExpr *expr) {
        UNREACHABLE("unsupported CXXThisExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitCXXThrowExpr(clang::CXXThrowExpr *expr) {
        UNREACHABLE("unsupported CXXThrowExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitCXXTypeidExpr(clang::CXXTypeidExpr *expr) {
        UNREACHABLE("unsupported CXXTypeidExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitCXXFoldExpr(clang::CXXFoldExpr *expr) {
        UNREACHABLE("unsupported CXXFoldExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitCXXUnresolvedConstructExpr(
        clang::CXXUnresolvedConstructExpr *expr) {
        UNREACHABLE("unsupported CXXUnresolvedConstructExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitCXXUuidofExpr(clang::CXXUuidofExpr *expr) {
        UNREACHABLE("unsupported CXXUuidofExpr");
    }

    mlir::FuncOp CodeGenVisitor::VisitDirectCallee(clang::FunctionDecl *callee) {
        auto name = callee->getName();
        return ctx.lookup_function(name);
    }

    mlir::Value CodeGenVisitor::VisitIndirectCallee(clang::Expr *callee) {
        return std::get< Value >(CodeGenVisitor::Visit(callee));
    }

    using Arguments = llvm::SmallVector< Value, 2 >;

    Arguments CodeGenVisitor::VisitArguments(clang::CallExpr *expr) {
        Arguments args;
        for (const auto &arg : expr->arguments()) {
            args.push_back(std::get< Value >(CodeGenVisitor::Visit(arg)));
        }
        return args;
    }

    ValueOrStmt CodeGenVisitor::VisitDirectCall(clang::CallExpr *expr) {
        auto loc    = builder.get_location(expr->getSourceRange());
        auto callee = CodeGenVisitor::VisitDirectCallee(expr->getDirectCallee());
        auto args   = CodeGenVisitor::VisitArguments(expr);
        return builder.make_value< CallOp >(loc, callee, args);
    }

    ValueOrStmt CodeGenVisitor::VisitIndirectCall(clang::CallExpr *expr) {
        auto loc    = builder.get_location(expr->getSourceRange());
        auto callee = CodeGenVisitor::VisitIndirectCallee(expr->getCallee());
        auto args   = CodeGenVisitor::VisitArguments(expr);
        return builder.make_value< IndirectCallOp >(loc, callee, args);
    }

    ValueOrStmt CodeGenVisitor::VisitCallExpr(clang::CallExpr *expr) {
        if (expr->getDirectCallee()) {
            return CodeGenVisitor::VisitDirectCall(expr);
        }

        return CodeGenVisitor::VisitIndirectCall(expr);
    }

    ValueOrStmt CodeGenVisitor::VisitCUDAKernelCallExpr(clang::CUDAKernelCallExpr *expr) {
        UNREACHABLE("unsupported CUDAKernelCallExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitCXXMemberCallExpr(clang::CXXMemberCallExpr *expr) {
        UNREACHABLE("unsupported CXXMemberCallExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitCXXOperatorCallExpr(clang::CXXOperatorCallExpr *expr) {
        UNREACHABLE("unsupported CXXOperatorCallExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitUserDefinedLiteral(clang::UserDefinedLiteral *lit) {
        UNREACHABLE("unsupported UserDefinedLiteral");
    }

    ValueOrStmt CodeGenVisitor::VisitCStyleCastExpr(clang::CStyleCastExpr *expr) {
        return make_cast< CStyleCastOp >(expr);
    }

    ValueOrStmt CodeGenVisitor::VisitCXXFunctionalCastExpr(clang::CXXFunctionalCastExpr *expr) {
        UNREACHABLE("unsupported CXXFunctionalCastExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitCXXConstCastExpr(clang::CXXConstCastExpr *expr) {
        UNREACHABLE("unsupported CXXConstCastExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitCXXDynamicCastExpr(clang::CXXDynamicCastExpr *expr) {
        UNREACHABLE("unsupported CXXDynamicCastExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitCXXReinterpretCastExpr(clang::CXXReinterpretCastExpr *expr) {
        UNREACHABLE("unsupported CXXReinterpretCastExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitCXXStaticCastExpr(clang::CXXStaticCastExpr *expr) {
        UNREACHABLE("unsupported CXXStaticCastExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitObjCBridgedCastExpr(clang::ObjCBridgedCastExpr *expr) {
        UNREACHABLE("unsupported ObjCBridgedCastExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitImplicitCastExpr(clang::ImplicitCastExpr *expr) {
        return make_cast< ImplicitCastOp >(expr);
    }

    ValueOrStmt CodeGenVisitor::VisitCharacterLiteral(clang::CharacterLiteral *lit) {
        return make_scalar_literal(lit);
    }

    ValueOrStmt CodeGenVisitor::VisitChooseExpr(clang::ChooseExpr *expr) {
        UNREACHABLE("unsupported ChooseExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitCompoundLiteralExpr(clang::CompoundLiteralExpr *expr) {
        UNREACHABLE("unsupported CompoundLiteralExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitConvertVectorExpr(clang::ConvertVectorExpr *expr) {
        UNREACHABLE("unsupported ConvertVectorExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitDeclRefExpr(clang::DeclRefExpr *expr) {
        auto loc = builder.get_location(expr->getSourceRange());

        // TODO(Heno): deal with function declaration

        // TODO(Heno): deal with enum constant declaration

        auto named = expr->getDecl()->getUnderlyingDecl();
        auto rty   = types.convert(expr->getType());
        return builder.make_value< DeclRefOp >(loc, rty, named->getNameAsString());
    }

    ValueOrStmt CodeGenVisitor::VisitDependentScopeDeclRefExpr(
        clang::DependentScopeDeclRefExpr *expr) {
        UNREACHABLE("unsupported DependentScopeDeclRefExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitDesignatedInitExpr(clang::DesignatedInitExpr *expr) {
        UNREACHABLE("unsupported DesignatedInitExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitExprWithCleanups(clang::ExprWithCleanups *expr) {
        UNREACHABLE("unsupported ExprWithCleanups");
    }

    ValueOrStmt CodeGenVisitor::VisitExpressionTraitExpr(clang::ExpressionTraitExpr *expr) {
        UNREACHABLE("unsupported ExpressionTraitExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitExtVectorElementExpr(clang::ExtVectorElementExpr *expr) {
        UNREACHABLE("unsupported ExtVectorElementExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitFloatingLiteral(clang::FloatingLiteral *lit) {
        return make_scalar_literal(lit);
    }

    ValueOrStmt CodeGenVisitor::VisitFunctionParmPackExpr(clang::FunctionParmPackExpr *expr) {
        UNREACHABLE("unsupported FunctionParmPackExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitGNUNullExpr(clang::GNUNullExpr *expr) {
        UNREACHABLE("unsupported GNUNullExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitGenericSelectionExpr(clang::GenericSelectionExpr *expr) {
        UNREACHABLE("unsupported GenericSelectionExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitImaginaryLiteral(clang::ImaginaryLiteral *lit) {
        UNREACHABLE("unsupported ImaginaryLiteral");
    }

    ValueOrStmt CodeGenVisitor::VisitFixedPointLiteral(clang::FixedPointLiteral *lit) {
        UNREACHABLE("unsupported FixedPointLiteral");
    }

    ValueOrStmt CodeGenVisitor::VisitImplicitValueInitExpr(clang::ImplicitValueInitExpr *expr) {
        UNREACHABLE("unsupported ImplicitValueInitExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitInitListExpr(clang::InitListExpr *expr) {
        auto loc = builder.get_location(expr->getSourceRange());
        auto ty  = types.convert(expr->getType());

        llvm::SmallVector< Value > elements;
        for (auto elem : expr->inits()) {
            elements.push_back(std::get< Value >(CodeGenVisitor::Visit(elem)));
        }

        return builder.make< InitListExpr >(loc, ty, elements);
    }

    ValueOrStmt CodeGenVisitor::VisitIntegerLiteral(const clang::IntegerLiteral *lit) {
        return make_scalar_literal(lit);
    }

    ValueOrStmt CodeGenVisitor::VisitLambdaExpr(clang::LambdaExpr *expr) {
        UNREACHABLE("unsupported LambdaExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitMSPropertyRefExpr(clang::MSPropertyRefExpr *expr) {
        UNREACHABLE("unsupported MSPropertyRefExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitMaterializeTemporaryExpr(
        clang::MaterializeTemporaryExpr *expr) {
        UNREACHABLE("unsupported MaterializeTemporaryExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitMemberExpr(clang::MemberExpr *expr) {
        auto loc   = builder.get_location(expr->getSourceRange());
        auto field = llvm::dyn_cast< clang::FieldDecl >(expr->getMemberDecl());
        auto name  = field->getName();
        auto base  = CodeGenVisitor::Visit(expr->getBase());
        auto type  = types.convert(expr->getType());
        return builder.make_value< RecordMemberOp >(loc, type, base, name);
    }

    ValueOrStmt CodeGenVisitor::VisitObjCArrayLiteral(clang::ObjCArrayLiteral *expr) {
        UNREACHABLE("unsupported ObjCArrayLiteral");
    }

    ValueOrStmt CodeGenVisitor::VisitObjCBoolLiteralExpr(clang::ObjCBoolLiteralExpr *expr) {
        UNREACHABLE("unsupported ObjCBoolLiteralExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitObjCBoxedExpr(clang::ObjCBoxedExpr *expr) {
        UNREACHABLE("unsupported ObjCBoxedExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitObjCDictionaryLiteral(clang::ObjCDictionaryLiteral *lit) {
        UNREACHABLE("unsupported ObjCDictionaryLiteral");
    }

    ValueOrStmt CodeGenVisitor::VisitObjCEncodeExpr(clang::ObjCEncodeExpr *expr) {
        UNREACHABLE("unsupported ObjCEncodeExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitObjCIndirectCopyRestoreExpr(
        clang::ObjCIndirectCopyRestoreExpr *expr) {
        UNREACHABLE("unsupported ObjCIndirectCopyRestoreExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitObjCIsaExpr(clang::ObjCIsaExpr *expr) {
        UNREACHABLE("unsupported ObjCIsaExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitObjCIvarRefExpr(clang::ObjCIvarRefExpr *expr) {
        UNREACHABLE("unsupported ObjCIvarRefExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitObjCMessageExpr(clang::ObjCMessageExpr *expr) {
        UNREACHABLE("unsupported ObjCMessageExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitObjCPropertyRefExpr(clang::ObjCPropertyRefExpr *expr) {
        UNREACHABLE("unsupported ObjCPropertyRefExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitObjCProtocolExpr(clang::ObjCProtocolExpr *expr) {
        UNREACHABLE("unsupported ObjCProtocolExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitObjCSelectorExpr(clang::ObjCSelectorExpr *expr) {
        UNREACHABLE("unsupported ObjCSelectorExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitObjCStringLiteral(clang::ObjCStringLiteral *lit) {
        UNREACHABLE("unsupported ObjCStringLiteral");
    }

    ValueOrStmt CodeGenVisitor::VisitObjCSubscriptRefExpr(clang::ObjCSubscriptRefExpr *expr) {
        UNREACHABLE("unsupported ObjCSubscriptRefExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitOffsetOfExpr(clang::OffsetOfExpr *expr) {
        UNREACHABLE("unsupported OffsetOfExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitOpaqueValueExpr(clang::OpaqueValueExpr *expr) {
        UNREACHABLE("unsupported OpaqueValueExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitOverloadExpr(clang::OverloadExpr *expr) {
        UNREACHABLE("unsupported OverloadExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitUnresolvedLookupExpr(clang::UnresolvedLookupExpr *expr) {
        UNREACHABLE("unsupported UnresolvedLookupExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitUnresolvedMemberExpr(clang::UnresolvedMemberExpr *expr) {
        UNREACHABLE("unsupported UnresolvedMemberExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitPackExpansionExpr(clang::PackExpansionExpr *expr) {
        UNREACHABLE("unsupported PackExpansionExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitParenExpr(clang::ParenExpr *expr) {
        auto loc     = builder.get_location(expr->getSourceRange());
        auto rty     = types.convert(expr->getType());
        auto subexpr = make_value_builder(expr->getSubExpr());
        return builder.make_value< ExprOp >(loc, rty, subexpr);
    }

    ValueOrStmt CodeGenVisitor::VisitParenListExpr(clang::ParenListExpr *expr) {
        UNREACHABLE("unsupported ParenListExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitPredefinedExpr(clang::PredefinedExpr *expr) {
        UNREACHABLE("unsupported PredefinedExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitPseudoObjectExpr(clang::PseudoObjectExpr *expr) {
        UNREACHABLE("unsupported PseudoObjectExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitShuffleVectorExpr(clang::ShuffleVectorExpr *expr) {
        UNREACHABLE("unsupported ShuffleVectorExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitSizeOfPackExpr(clang::SizeOfPackExpr *expr) {
        UNREACHABLE("unsupported SizeOfPackExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitStmtExpr(clang::StmtExpr *expr) {
        UNREACHABLE("unsupported StmtExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitStringLiteral(clang::StringLiteral *lit) {
        auto type = types.convert(lit->getType());
        auto loc  = builder.get_location(lit->getSourceRange());
        return builder.constant(loc, type, lit->getString());
    }

    ValueOrStmt CodeGenVisitor::VisitSubstNonTypeTemplateParmExpr(
        clang::SubstNonTypeTemplateParmExpr *expr) {
        UNREACHABLE("unsupported SubstNonTypeTemplateParmExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitSubstNonTypeTemplateParmPackExpr(
        clang::SubstNonTypeTemplateParmPackExpr *expr) {
        UNREACHABLE("unsupported SubstNonTypeTemplateParmPackExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitTypeTraitExpr(clang::TypeTraitExpr *expr) {
        UNREACHABLE("unsupported TypeTraitExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitUnaryExprOrTypeTraitExpr(
        clang::UnaryExprOrTypeTraitExpr *expr) {
        auto kind = expr->getKind();

        if (kind == clang::UETT_SizeOf) {
            return dispatch_trait_expr< SizeOfTypeOp, SizeOfExprOp >(expr);
        }

        if (kind == clang::UETT_AlignOf) {
            return dispatch_trait_expr< AlignOfTypeOp, AlignOfExprOp >(expr);
        }

        UNREACHABLE("unsupported UnaryExprOrTypeTraitExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitSourceLocExpr(clang::SourceLocExpr *expr) {
        UNREACHABLE("unsupported SourceLocExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitVAArgExpr(clang::VAArgExpr *expr) {
        UNREACHABLE("unsupported VAArgExpr");
    }

    // Statements

    ValueOrStmt CodeGenVisitor::VisitForStmt(clang::ForStmt *stmt) {
        auto loc = builder.get_location(stmt->getSourceRange());

        auto init_builder = make_region_builder(stmt->getInit());
        auto incr_builder = make_region_builder(stmt->getInc());
        auto body_builder = make_region_builder(stmt->getBody());

        if (auto cond = stmt->getCond()) {
            auto cond_builder = make_cond_builder(cond);
            return builder.make< ForOp >(
                loc, init_builder, cond_builder, incr_builder, body_builder);
        }
        return builder.make< ForOp >(
            loc, init_builder, make_yield_true(), incr_builder, body_builder);
    }

    ValueOrStmt CodeGenVisitor::VisitGotoStmt(clang::GotoStmt *stmt) {
        UNREACHABLE("unsupported GotoStmt");
    }

    ValueOrStmt CodeGenVisitor::VisitIfStmt(clang::IfStmt *stmt) {
        auto loc = builder.get_location(stmt->getSourceRange());

        auto cond_builder = make_cond_builder(stmt->getCond());
        auto then_builder = make_region_builder(stmt->getThen());

        if (stmt->getElse()) {
            return builder.make< IfOp >(
                loc, cond_builder, then_builder, make_region_builder(stmt->getElse()));
        }
        return builder.make< IfOp >(loc, cond_builder, then_builder);
    }

    ValueOrStmt CodeGenVisitor::VisitIndirectGotoStmt(clang::IndirectGotoStmt *stmt) {
        UNREACHABLE("unsupported IndirectGotoStmt");
    }

    ValueOrStmt CodeGenVisitor::VisitLabelStmt(clang::LabelStmt *stmt) {
        UNREACHABLE("unsupported LabelStmt");
    }

    ValueOrStmt CodeGenVisitor::VisitMSDependentExistsStmt(clang::MSDependentExistsStmt *stmt) {
        UNREACHABLE("unsupported MSDependentExistsStmt");
    }

    ValueOrStmt CodeGenVisitor::VisitNullStmt(clang::NullStmt *stmt) {
        UNREACHABLE("unsupported NullStmt");
    }

    ValueOrStmt CodeGenVisitor::VisitOMPBarrierDirective(clang::OMPBarrierDirective *dir) {
        UNREACHABLE("unsupported OMPBarrierDirective");
    }

    ValueOrStmt CodeGenVisitor::VisitOMPCriticalDirective(clang::OMPCriticalDirective *dir) {
        UNREACHABLE("unsupported OMPCriticalDirective");
    }

    ValueOrStmt CodeGenVisitor::VisitOMPFlushDirective(clang::OMPFlushDirective *dir) {
        UNREACHABLE("unsupported OMPFlushDirective");
    }

    ValueOrStmt CodeGenVisitor::VisitOMPForDirective(clang::OMPForDirective *dir) {
        UNREACHABLE("unsupported OMPForDirective");
    }

    ValueOrStmt CodeGenVisitor::VisitOMPMasterDirective(clang::OMPMasterDirective *dir) {
        UNREACHABLE("unsupported OMPMasterDirective");
    }

    ValueOrStmt CodeGenVisitor::VisitOMPParallelDirective(clang::OMPParallelDirective *dir) {
        UNREACHABLE("unsupported OMPParallelDirective");
    }

    ValueOrStmt CodeGenVisitor::VisitOMPParallelForDirective(clang::OMPParallelForDirective *dir) {
        UNREACHABLE("unsupported OMPParallelForDirective");
    }

    ValueOrStmt CodeGenVisitor::VisitOMPParallelSectionsDirective(
        clang::OMPParallelSectionsDirective *dir) {
        UNREACHABLE("unsupported OMPParallelSectionsDirective");
    }

    ValueOrStmt CodeGenVisitor::VisitOMPSectionDirective(clang::OMPSectionDirective *dir) {
        UNREACHABLE("unsupported OMPSectionDirective");
    }

    ValueOrStmt CodeGenVisitor::VisitOMPSectionsDirective(clang::OMPSectionsDirective *dir) {
        UNREACHABLE("unsupported OMPSectionsDirective");
    }

    ValueOrStmt CodeGenVisitor::VisitOMPSimdDirective(clang::OMPSimdDirective *dir) {
        UNREACHABLE("unsupported OMPSimdDirective");
    }

    ValueOrStmt CodeGenVisitor::VisitOMPSingleDirective(clang::OMPSingleDirective *dir) {
        UNREACHABLE("unsupported OMPSingleDirective");
    }

    ValueOrStmt CodeGenVisitor::VisitOMPTaskDirective(clang::OMPTaskDirective *dir) {
        UNREACHABLE("unsupported OMPTaskDirective");
    }

    ValueOrStmt CodeGenVisitor::VisitOMPTaskwaitDirective(clang::OMPTaskwaitDirective *dir) {
        UNREACHABLE("unsupported OMPTaskwaitDirective");
    }

    ValueOrStmt CodeGenVisitor::VisitOMPTaskyieldDirective(clang::OMPTaskyieldDirective *dir) {
        UNREACHABLE("unsupported OMPTaskyieldDirective");
    }

    ValueOrStmt CodeGenVisitor::VisitObjCAtCatchStmt(clang::ObjCAtCatchStmt *stmt) {
        UNREACHABLE("unsupported ObjCAtCatchStmt");
    }

    ValueOrStmt CodeGenVisitor::VisitObjCAtFinallyStmt(clang::ObjCAtFinallyStmt *stmt) {
        UNREACHABLE("unsupported ObjCAtFinallyStmt");
    }

    ValueOrStmt CodeGenVisitor::VisitObjCAtSynchronizedStmt(clang::ObjCAtSynchronizedStmt *stmt) {
        UNREACHABLE("unsupported ObjCAtSynchronizedStmt");
    }

    ValueOrStmt CodeGenVisitor::VisitObjCAtThrowStmt(clang::ObjCAtThrowStmt *stmt) {
        UNREACHABLE("unsupported ObjCAtThrowStmt");
    }

    ValueOrStmt CodeGenVisitor::VisitObjCAtTryStmt(clang::ObjCAtTryStmt *stmt) {
        UNREACHABLE("unsupported ObjCAtTryStmt");
    }

    ValueOrStmt CodeGenVisitor::VisitObjCAutoreleasePoolStmt(clang::ObjCAutoreleasePoolStmt *stmt) {
        UNREACHABLE("unsupported ObjCAutoreleasePoolStmt");
    }

    ValueOrStmt CodeGenVisitor::VisitObjCForCollectionStmt(clang::ObjCForCollectionStmt *stmt) {
        UNREACHABLE("unsupported ObjCForCollectionStmt");
    }

    ValueOrStmt CodeGenVisitor::VisitReturnStmt(clang::ReturnStmt *stmt) {
        auto loc = builder.get_location(stmt->getSourceRange());
        if (auto ret = stmt->getRetValue())
            return builder.make< ReturnOp >(loc, CodeGenVisitor::Visit(ret));
        return builder.make< ReturnOp >(loc);
    }

    ValueOrStmt CodeGenVisitor::VisitSEHExceptStmt(clang::SEHExceptStmt *stmt) {
        UNREACHABLE("unsupported SEHExceptStmt");
    }

    ValueOrStmt CodeGenVisitor::VisitSEHFinallyStmt(clang::SEHFinallyStmt *stmt) {
        UNREACHABLE("unsupported SEHFinallyStmt");
    }

    ValueOrStmt CodeGenVisitor::VisitSEHLeaveStmt(clang::SEHLeaveStmt *stmt) {
        UNREACHABLE("unsupported SEHLeaveStmt");
    }

    ValueOrStmt CodeGenVisitor::VisitSEHTryStmt(clang::SEHTryStmt *stmt) {
        UNREACHABLE("unsupported SEHTryStmt");
    }

    ValueOrStmt CodeGenVisitor::VisitCaseStmt(clang::CaseStmt *stmt) {
        auto loc          = builder.get_location(stmt->getSourceRange());
        auto lhs_builder  = make_value_builder(stmt->getLHS());
        auto body_builder = make_region_builder(stmt->getSubStmt());
        return builder.make< CaseOp >(loc, lhs_builder, body_builder);
    }

    ValueOrStmt CodeGenVisitor::VisitDefaultStmt(clang::DefaultStmt *stmt) {
        auto loc          = builder.get_location(stmt->getSourceRange());
        auto body_builder = make_region_builder(stmt->getSubStmt());
        return builder.make< DefaultOp >(loc, body_builder);
    }

    ValueOrStmt CodeGenVisitor::VisitSwitchStmt(clang::SwitchStmt *stmt) {
        auto loc          = builder.get_location(stmt->getSourceRange());
        auto cond_builder = make_value_builder(stmt->getCond());
        auto body_builder = make_region_builder(stmt->getBody());
        if (stmt->getInit()) {
            return builder.make< SwitchOp >(
                loc, make_region_builder(stmt->getInit()), cond_builder, body_builder);
        }
        return builder.make< SwitchOp >(loc, nullptr, cond_builder, body_builder);
    }

    ValueOrStmt CodeGenVisitor::VisitWhileStmt(clang::WhileStmt *stmt) {
        auto loc          = builder.get_location(stmt->getSourceRange());
        auto cond_builder = make_cond_builder(stmt->getCond());
        auto body_builder = make_region_builder(stmt->getBody());
        return builder.make< WhileOp >(loc, cond_builder, body_builder);
    }

    ValueOrStmt CodeGenVisitor::VisitBuiltinBitCastExpr(clang::BuiltinBitCastExpr *expr) {
        return make_cast< BuiltinBitCastOp >(expr);
    }

    // Declarations

    ValueOrStmt CodeGenVisitor::VisitImportDecl(clang::ImportDecl *decl) {
        UNREACHABLE("unsupported ImportDecl");
    }

    ValueOrStmt CodeGenVisitor::VisitEmptyDecl(clang::EmptyDecl *decl) {
        UNREACHABLE("unsupported EmptyDecl");
    }

    ValueOrStmt CodeGenVisitor::VisitAccessSpecDecl(clang::AccessSpecDecl *decl) {
        UNREACHABLE("unsupported AccessSpecDecl");
    }

    ValueOrStmt CodeGenVisitor::VisitCapturedDecl(clang::CapturedDecl *decl) {
        UNREACHABLE("unsupported CapturedDecl");
    }

    ValueOrStmt CodeGenVisitor::VisitClassScopeFunctionSpecializationDecl(
        clang::ClassScopeFunctionSpecializationDecl *decl) {
        UNREACHABLE("unsupported ClassScopeFunctionSpecializationDecl");
    }

    ValueOrStmt CodeGenVisitor::VisitExportDecl(clang::ExportDecl *decl) {
        UNREACHABLE("unsupported ExportDecl");
    }

    ValueOrStmt CodeGenVisitor::VisitExternCContextDecl(clang::ExternCContextDecl *decl) {
        UNREACHABLE("unsupported ExternCContextDecl");
    }

    ValueOrStmt CodeGenVisitor::VisitFileScopeAsmDecl(clang::FileScopeAsmDecl *decl) {
        UNREACHABLE("unsupported FileScopeAsmDecl");
    }

    ValueOrStmt CodeGenVisitor::VisitStaticAssertDecl(clang::StaticAssertDecl *decl) {
        UNREACHABLE("unsupported StaticAssertDecl");
    }

    ValueOrStmt CodeGenVisitor::VisitTranslationUnitDecl(clang::TranslationUnitDecl *decl) {
        UNREACHABLE("unsupported TranslationUnitDecl");
    }

    ValueOrStmt CodeGenVisitor::VisitBindingDecl(clang::BindingDecl *decl) {
        UNREACHABLE("unsupported BindingDecl");
    }

    // ValueOrStmt CodeGenVisitor::VisitNamespaceDecl(clang::NamespaceDecl *decl)
    // {
    //     UNREACHABLE( "unsupported NamespaceDecl" );
    // }

    ValueOrStmt CodeGenVisitor::VisitNamespaceAliasDecl(clang::NamespaceAliasDecl *decl) {
        UNREACHABLE("unsupported NamespaceAliasDecl");
    }

    // ValueOrStmt CodeGenVisitor::VisitTypedefNameDecl(clang::TypedefNameDecl *decl)
    // {
    //     UNREACHABLE( "unsupported TypedefNameDecl" );
    // }

    ValueOrStmt CodeGenVisitor::VisitTypedefDecl(clang::TypedefDecl *decl) {
        auto loc  = builder.get_location(decl->getSourceRange());
        auto name = decl->getName();

        auto type = [&]() -> mlir::Type {
            auto underlying = decl->getUnderlyingType();
            if (auto fty = clang::dyn_cast< clang::FunctionType >(underlying)) {
                return types.convert(fty);
            }

            // predeclare named underlying types if necessery
            walk_type(underlying, [&](auto ty) {
                if (auto tag = clang::dyn_cast< clang::TagType >(ty)) {
                    CodeGenVisitor::Visit(tag->getDecl());
                    return true; // stop recursive walk
                }

                return false;
            });

            return types.convert(underlying);
        }();

        return builder.define_type(loc, type, name);
    }

    ValueOrStmt CodeGenVisitor::VisitTypeAliasDecl(clang::TypeAliasDecl *decl) {
        UNREACHABLE("unsupported TypeAliasDecl");
    }
    ValueOrStmt CodeGenVisitor::VisitTemplateDecl(clang::TemplateDecl *decl) {
        UNREACHABLE("unsupported TemplateDecl");
    }

    ValueOrStmt CodeGenVisitor::VisitTypeAliasTemplateDecl(clang::TypeAliasTemplateDecl *decl) {
        UNREACHABLE("unsupported TypeAliasTemplateDecl");
    }

    ValueOrStmt CodeGenVisitor::VisitLabelDecl(clang::LabelDecl *decl) {
        UNREACHABLE("unsupported LabelDecl");
    }

    ValueOrStmt CodeGenVisitor::VisitEnumDecl(clang::EnumDecl *decl) {
        auto loc  = builder.get_location(decl->getSourceRange());
        auto name = ctx.elaborated_name(decl);
        auto type = types.convert(decl->getIntegerType());

        auto constants = [&](auto &bld, auto loc) {
            for (auto con : decl->enumerators()) {
                CodeGenVisitor::Visit(con);
            }
        };

        return builder.declare_enum(loc, name, type, constants);
    }

    ValueOrStmt CodeGenVisitor::VisitRecordDecl(clang::RecordDecl *decl) {
        auto loc  = builder.get_location(decl->getSourceRange());
        auto name = ctx.elaborated_name(decl);
        // declare the type first to allow recursive type definitions
        auto rec_decl = builder.declare_type(loc, name);
        if (!decl->isCompleteDefinition()) {
            return rec_decl;
        }

        // generate record definition
        if (decl->field_empty()) {
            return builder.make< RecordDeclOp >(loc, name);
        }

        return builder.make< RecordDeclOp >(loc, name, [&](auto &bld, auto loc) {
            for (auto field : decl->fields()) {
                CodeGenVisitor::Visit(field);
            }
        });
    }

    ValueOrStmt CodeGenVisitor::VisitEnumConstantDecl(clang::EnumConstantDecl *decl) {
        auto loc   = builder.get_location(decl->getSourceRange());
        auto name  = decl->getName();
        auto value = decl->getInitVal();

        if (decl->getInitExpr()) {
            auto init = make_value_builder(decl->getInitExpr());
            return builder.make< EnumConstantOp >(loc, name, value, init);
        }

        return builder.make< EnumConstantOp >(loc, name, value);
    }

    ValueOrStmt CodeGenVisitor::VisitFunctionDecl(clang::FunctionDecl *decl) {
        auto name = decl->getName();

        if (auto fn = ctx.functions.lookup(name))
            return fn;

        ScopedInsertPoint builder_scope(builder);
        llvm::ScopedHashTableScope scope(ctx.variables);

        auto loc  = builder.get_location(decl->getSourceRange());
        auto type = types.convert(decl->getFunctionType());
        assert(type);

        auto fn = builder.make< mlir::FuncOp >(loc, name, type);
        if (failed(ctx.functions.declare(name, fn))) {
            ctx.error("error: multiple declarations of a same function" + name);
        }

        // TODO(Heno): move to function prototype lifting
        if (!decl->isMain())
            fn.setVisibility(mlir::FuncOp::Visibility::Private);

        if (!decl->hasBody() || !fn.isExternal())
            return Value(); // dummy value

        auto entry = fn.addEntryBlock();

        // In MLIR the entry block of the function must have the same argument list as the
        // function itself.
        for (const auto &[arg, earg] : llvm::zip(decl->parameters(), entry->getArguments())) {
            if (failed(ctx.variables.declare(arg->getName(), earg)))
                ctx.error("error: multiple declarations of a same symbol" + arg->getName());
        }

        builder.set_insertion_point_to_start(entry);

        // emit function body
        if (decl->hasBody()) {
            CodeGenVisitor::Visit(decl->getBody());
        }

        splice_trailing_scopes(fn);

        auto &last_block = fn.getBlocks().back();
        auto &ops        = last_block.getOperations();
        builder.set_insertion_point_to_end(&last_block);

        if (ops.empty() || !ops.back().hasTrait< mlir::OpTrait::IsTerminator >()) {
            auto beg_loc = builder.get_location(decl->getBeginLoc());
            auto end_loc = builder.get_location(decl->getEndLoc());
            if (decl->getReturnType()->isVoidType()) {
                builder.make< ReturnOp >(end_loc);
            } else {
                if (decl->isMain()) {
                    // return zero if no return is present in main
                    auto zero = builder.constant(end_loc, type.getResult(0), apint(0));
                    builder.make< ReturnOp >(end_loc, zero);
                } else {
                    builder.make< UnreachableOp >(beg_loc);
                }
            }
        }

        return fn;
    }

    ValueOrStmt CodeGenVisitor::VisitCXXMethodDecl(clang::CXXMethodDecl *decl) {
        UNREACHABLE("unsupported CXXMethodDecl");
    }

    ValueOrStmt CodeGenVisitor::VisitCXXConstructorDecl(clang::CXXConstructorDecl *decl) {
        UNREACHABLE("unsupported CXXConstructorDecl");
    }

    ValueOrStmt CodeGenVisitor::VisitCXXDestructorDecl(clang::CXXDestructorDecl *decl) {
        UNREACHABLE("unsupported CXXDestructorDecl");
    }

    ValueOrStmt CodeGenVisitor::VisitCXXConversionDecl(clang::CXXConversionDecl *decl) {
        UNREACHABLE("unsupported CXXConversionDecl");
    }

    ValueOrStmt CodeGenVisitor::VisitCXXDeductionGuideDecl(clang::CXXDeductionGuideDecl *decl) {
        UNREACHABLE("unsupported CXXDeductionGuideDecl");
    }

    ValueOrStmt CodeGenVisitor::VisitMSPropertyDecl(clang::MSPropertyDecl *decl) {
        UNREACHABLE("unsupported MSPropertyDecl");
    }

    ValueOrStmt CodeGenVisitor::VisitMSGuidDecl(clang::MSGuidDecl *decl) {
        UNREACHABLE("unsupported MSGuidDecl");
    }

    ValueOrStmt CodeGenVisitor::VisitFieldDecl(clang::FieldDecl *decl) {
        auto loc  = builder.get_location(decl->getSourceRange());
        auto name = decl->getName();
        auto type = types.convert(decl->getType());
        return builder.make< FieldDeclOp >(loc, name, type);
    }

    ValueOrStmt CodeGenVisitor::VisitIndirectFieldDecl(clang::IndirectFieldDecl *decl) {
        UNREACHABLE("unsupported IndirectFieldDecl");
    }

    ValueOrStmt CodeGenVisitor::VisitFriendDecl(clang::FriendDecl *decl) {
        UNREACHABLE("unsupported FriendDecl");
    }

    ValueOrStmt CodeGenVisitor::VisitFriendTemplateDecl(clang::FriendTemplateDecl *decl) {
        UNREACHABLE("unsupported FriendTemplateDecl");
    }

    ValueOrStmt CodeGenVisitor::VisitObjCAtDefsFieldDecl(clang::ObjCAtDefsFieldDecl *decl) {
        UNREACHABLE("unsupported ObjCAtDefsFieldDecl");
    }

    ValueOrStmt CodeGenVisitor::VisitObjCIvarDecl(clang::ObjCIvarDecl *decl) {
        UNREACHABLE("unsupported ObjCIvarDecl");
    }

    ValueOrStmt CodeGenVisitor::VisitVarDecl(clang::VarDecl *decl) {
        auto ty   = types.convert(decl->getType());
        auto name = decl->getUnderlyingDecl()->getName();
        auto loc  = builder.get_end_location(decl->getSourceRange());

        if (decl->getInit()) {
            auto init = make_value_builder(decl->getInit());
            return make_vardecl(decl, loc, ty, name, init);
        }

        return make_vardecl(decl, loc, ty, name);
    }

    ValueOrStmt CodeGenVisitor::VisitDecompositionDecl(clang::DecompositionDecl *decl) {
        UNREACHABLE("unsupported DecompositionDecl");
    }

    ValueOrStmt CodeGenVisitor::VisitImplicitParamDecl(clang::ImplicitParamDecl *decl) {
        UNREACHABLE("unsupported ImplicitParamDecl");
    }

    // ValueOrStmt
    // CodeGenVisitor::VisitUnresolvedUsingIfExistsDecl(clang::UnresolvedUsingIfExistsDecl
    // *decl)
    // {
    //     UNREACHABLE( "unsupported UnresolvedUsingIfExistsDecl" );
    // }

    ValueOrStmt CodeGenVisitor::VisitParmVarDecl(clang::ParmVarDecl *decl) {
        UNREACHABLE("unsupported ParmVarDecl");
    }

    ValueOrStmt CodeGenVisitor::VisitObjCMethodDecl(clang::ObjCMethodDecl *decl) {
        UNREACHABLE("unsupported ObjCMethodDecl");
    }

    ValueOrStmt CodeGenVisitor::VisitObjCTypeParamDecl(clang::ObjCTypeParamDecl *decl) {
        UNREACHABLE("unsupported ObjCTypeParamDecl");
    }

    ValueOrStmt CodeGenVisitor::VisitObjCProtocolDecl(clang::ObjCProtocolDecl *decl) {
        UNREACHABLE("unsupported ObjCProtocolDecl");
    }

    ValueOrStmt CodeGenVisitor::VisitLinkageSpecDecl(clang::LinkageSpecDecl *decl) {
        UNREACHABLE("unsupported LinkageSpecDecl");
    }

    ValueOrStmt CodeGenVisitor::VisitUsingDecl(clang::UsingDecl *decl) {
        UNREACHABLE("unsupported UsingDecl");
    }

    ValueOrStmt CodeGenVisitor::VisitUsingShadowDecl(clang::UsingShadowDecl *decl) {
        UNREACHABLE("unsupported UsingShadowDecl");
    }

    ValueOrStmt CodeGenVisitor::VisitUsingDirectiveDecl(clang::UsingDirectiveDecl *decl) {
        UNREACHABLE("unsupported UsingDirectiveDecl");
    }

    ValueOrStmt CodeGenVisitor::VisitUsingPackDecl(clang::UsingPackDecl *decl) {
        UNREACHABLE("unsupported UsingPackDecl");
    }

    // ValueOrStmt CodeGenVisitor::VisitUsingEnumDecl(clang::UsingEnumDecl *decl)
    // {
    //     UNREACHABLE( "unsupported UsingEnumDecl" );
    // }

    ValueOrStmt CodeGenVisitor::VisitUnresolvedUsingValueDecl(
        clang::UnresolvedUsingValueDecl *decl) {
        UNREACHABLE("unsupported UnresolvedUsingValueDecl");
    }

    ValueOrStmt CodeGenVisitor::VisitUnresolvedUsingTypenameDecl(
        clang::UnresolvedUsingTypenameDecl *decl) {
        UNREACHABLE("unsupported UnresolvedUsingTypenameDecl");
    }

    ValueOrStmt CodeGenVisitor::VisitBuiltinTemplateDecl(clang::BuiltinTemplateDecl *decl) {
        UNREACHABLE("unsupported BuiltinTemplateDecl");
    }

    ValueOrStmt CodeGenVisitor::VisitConceptDecl(clang::ConceptDecl *decl) {
        UNREACHABLE("unsupported ConceptDecl");
    }

    ValueOrStmt CodeGenVisitor::VisitRedeclarableTemplateDecl(
        clang::RedeclarableTemplateDecl *decl) {
        UNREACHABLE("unsupported RedeclarableTemplateDecl");
    }

    ValueOrStmt CodeGenVisitor::VisitLifetimeExtendedTemporaryDecl(
        clang::LifetimeExtendedTemporaryDecl *decl) {
        UNREACHABLE("unsupported LifetimeExtendedTemporaryDecl");
    }

    ValueOrStmt CodeGenVisitor::VisitPragmaCommentDecl(clang::PragmaCommentDecl *decl) {
        UNREACHABLE("unsupported PragmaCommentDecl");
    }

    ValueOrStmt CodeGenVisitor::VisitPragmaDetectMismatchDecl(
        clang::PragmaDetectMismatchDecl *decl) {
        UNREACHABLE("unsupported PragmaDetectMismatchDecl");
    }

    ValueOrStmt CodeGenVisitor::VisitRequiresExprBodyDecl(clang::RequiresExprBodyDecl *decl) {
        UNREACHABLE("unsupported RequiresExprBodyDecl");
    }

    ValueOrStmt CodeGenVisitor::VisitObjCCompatibleAliasDecl(clang::ObjCCompatibleAliasDecl *decl) {
        UNREACHABLE("unsupported ObjCCompatibleAliasDecl");
    }

    ValueOrStmt CodeGenVisitor::VisitObjCCategoryDecl(clang::ObjCCategoryDecl *decl) {
        UNREACHABLE("unsupported ObjCCategoryDecl");
    }

    ValueOrStmt CodeGenVisitor::VisitObjCImplDecl(clang::ObjCImplDecl *decl) {
        UNREACHABLE("unsupported ObjCImplDecl");
    }

    ValueOrStmt CodeGenVisitor::VisitObjCInterfaceDecl(clang::ObjCInterfaceDecl *decl) {
        UNREACHABLE("unsupported ObjCInterfaceDecl");
    }

    ValueOrStmt CodeGenVisitor::VisitObjCCategoryImplDecl(clang::ObjCCategoryImplDecl *decl) {
        UNREACHABLE("unsupported ObjCCategoryImplDecl");
    }

    ValueOrStmt CodeGenVisitor::VisitObjCImplementationDecl(clang::ObjCImplementationDecl *decl) {
        UNREACHABLE("unsupported ObjCImplementationDecl");
    }

    ValueOrStmt CodeGenVisitor::VisitObjCPropertyDecl(clang::ObjCPropertyDecl *decl) {
        UNREACHABLE("unsupported ObjCPropertyDecl");
    }

    ValueOrStmt CodeGenVisitor::VisitObjCPropertyImplDecl(clang::ObjCPropertyImplDecl *decl) {
        UNREACHABLE("unsupported ObjCPropertyImplDecl");
    }

    ValueOrStmt CodeGenVisitor::VisitTemplateParamObjectDecl(clang::TemplateParamObjectDecl *decl) {
        UNREACHABLE("unsupported TemplateParamObjectDecl");
    }

    ValueOrStmt CodeGenVisitor::VisitTemplateTypeParmDecl(clang::TemplateTypeParmDecl *decl) {
        UNREACHABLE("unsupported TemplateTypeParmDecl");
    }

    ValueOrStmt CodeGenVisitor::VisitNonTypeTemplateParmDecl(clang::NonTypeTemplateParmDecl *decl) {
        UNREACHABLE("unsupported NonTypeTemplateParmDecl");
    }

    ValueOrStmt CodeGenVisitor::VisitTemplateTemplateParmDecl(
        clang::TemplateTemplateParmDecl *decl) {
        UNREACHABLE("unsupported TemplateTemplateParmDecl");
    }

    ValueOrStmt CodeGenVisitor::VisitClassTemplateDecl(clang::ClassTemplateDecl *decl) {
        UNREACHABLE("unsupported ClassTemplateDecl");
    }

    ValueOrStmt CodeGenVisitor::VisitClassTemplatePartialSpecializationDecl(
        clang::ClassTemplatePartialSpecializationDecl *decl) {
        UNREACHABLE("unsupported ClassTemplatePartialSpecializationDecl");
    }

    ValueOrStmt CodeGenVisitor::VisitClassTemplateSpecializationDecl(
        clang::ClassTemplateSpecializationDecl *decl) {
        UNREACHABLE("unsupported ClassTemplateSpecializationDecl");
    }

    ValueOrStmt CodeGenVisitor::VisitVarTemplateDecl(clang::VarTemplateDecl *decl) {
        UNREACHABLE("unsupported VarTemplateDecl");
    }

    ValueOrStmt CodeGenVisitor::VisitVarTemplateSpecializationDecl(
        clang::VarTemplateSpecializationDecl *decl) {
        UNREACHABLE("unsupported VarTemplateSpecializationDecl");
    }

    ValueOrStmt CodeGenVisitor::VisitVarTemplatePartialSpecializationDecl(
        clang::VarTemplatePartialSpecializationDecl *decl) {
        UNREACHABLE("unsupported VarTemplatePartialSpecializationDecl");
    }

    ValueOrStmt CodeGenVisitor::VisitFunctionTemplateDecl(clang::FunctionTemplateDecl *decl) {
        UNREACHABLE("unsupported FunctionTemplateDecl");
    }

    ValueOrStmt CodeGenVisitor::VisitConstructorUsingShadowDecl(
        clang::ConstructorUsingShadowDecl *decl) {
        UNREACHABLE("unsupported ConstructorUsingShadowDecl");
    }

    ValueOrStmt CodeGenVisitor::VisitOMPAllocateDecl(clang::OMPAllocateDecl *decl) {
        UNREACHABLE("unsupported OMPAllocateDecl");
    }

    ValueOrStmt CodeGenVisitor::VisitOMPRequiresDecl(clang::OMPRequiresDecl *decl) {
        UNREACHABLE("unsupported OMPRequiresDecl");
    }

    ValueOrStmt CodeGenVisitor::VisitOMPThreadPrivateDecl(clang::OMPThreadPrivateDecl *decl) {
        UNREACHABLE("unsupported OMPThreadPrivateDecl");
    }

    ValueOrStmt CodeGenVisitor::VisitOMPCapturedExprDecl(clang::OMPCapturedExprDecl *decl) {
        UNREACHABLE("unsupported OMPCapturedExprDecl");
    }

    ValueOrStmt CodeGenVisitor::VisitOMPDeclareReductionDecl(clang::OMPDeclareReductionDecl *decl) {
        UNREACHABLE("unsupported OMPDeclareReductionDecl");
    }

    ValueOrStmt CodeGenVisitor::VisitOMPDeclareMapperDecl(clang::OMPDeclareMapperDecl *decl) {
        UNREACHABLE("unsupported OMPDeclareMapperDecl");
    }

} // namespace vast::hl
