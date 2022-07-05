// Copyright (c) 2022-present, Trail of Bits, Inc.

#include "vast/Translation/HighLevelVisitor.hpp"
#include "vast/Translation/DataLayout.hpp"

namespace vast::hl
{
    std::string get_field_name(clang::FieldDecl *decl) {
        if (decl->isAnonymousStructOrUnion())
            return "anonymous." + std::to_string(decl->getFieldIndex());
        return decl->getName().str();
    }

    // Binary Operations

    ValueOrStmt CodeGenVisitor::VisitBinPtrMemD(clang::BinaryOperator *expr) {
        VAST_UNREACHABLE("unsupported BinPtrMemD");
    }

    ValueOrStmt CodeGenVisitor::VisitBinPtrMemI(clang::BinaryOperator *expr) {
        VAST_UNREACHABLE("unsupported BinPtrMemI");
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
        return checked(make_cmp< Predicate::ult, Predicate::slt >(expr));
    }

    ValueOrStmt CodeGenVisitor::VisitBinGT(clang::BinaryOperator *expr) {
        return checked(make_cmp< Predicate::ugt, Predicate::sgt >(expr));
    }

    ValueOrStmt CodeGenVisitor::VisitBinLE(clang::BinaryOperator *expr) {
        return checked(make_cmp< Predicate::ule, Predicate::sle >(expr));
    }

    ValueOrStmt CodeGenVisitor::VisitBinGE(clang::BinaryOperator *expr) {
        return checked(make_cmp< Predicate::uge, Predicate::sge >(expr));
    }

    ValueOrStmt CodeGenVisitor::VisitBinEQ(clang::BinaryOperator *expr) {
        return checked(make_cmp< Predicate::eq >(expr));
    }

    ValueOrStmt CodeGenVisitor::VisitBinNE(clang::BinaryOperator *expr) {
        return checked(make_cmp< Predicate::ne >(expr));
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
        auto lhs = Visit(expr->getLHS());
        auto rhs = Visit(expr->getRHS());
        auto ty = types.convert(expr->getType());
        auto loc = builder.get_end_location(expr->getSourceRange());
        return builder.make_value< BinComma >(loc, ty, lhs, rhs);
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
        return make_unary_non_lvalue< AddressOf >(expr);
    }

    ValueOrStmt CodeGenVisitor::VisitUnaryDeref(clang::UnaryOperator *expr) {
        return make_unary_lvalue< Deref >(expr);
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
        VAST_UNREACHABLE("unsupported UnaryReal");
    }

    ValueOrStmt CodeGenVisitor::VisitUnaryImag(clang::UnaryOperator *expr) {
        VAST_UNREACHABLE("unsupported UnaryImag");
    }

    ValueOrStmt CodeGenVisitor::VisitUnaryExtension(clang::UnaryOperator *expr) {
        VAST_UNREACHABLE("unsupported UnaryExtension");
    }

    ValueOrStmt CodeGenVisitor::VisitUnaryCoawait(clang::UnaryOperator *expr) {
        VAST_UNREACHABLE("unsupported UnaryCoawait");
    }

    // Assembly Statements

    ValueOrStmt CodeGenVisitor::VisitAsmStmt(clang::AsmStmt *stmt) {
        VAST_UNREACHABLE("unsupported AsmStmt");
    }

    ValueOrStmt CodeGenVisitor::VisitGCCAsmStmt(clang::GCCAsmStmt *stmt) {
        VAST_UNREACHABLE("unsupported GCCAsmStmt");
    }

    ValueOrStmt CodeGenVisitor::VisitMSAsmStmt(clang::MSAsmStmt *stmt) {
        VAST_UNREACHABLE("unsupported MSAsmStmt");
    }

    ValueOrStmt CodeGenVisitor::VisitCoroutineBodyStmt(clang::CoroutineBodyStmt *stmt) {
        VAST_UNREACHABLE("unsupported CoroutineBodyStmt");
    }

    ValueOrStmt CodeGenVisitor::VisitCoreturnStmt(clang::CoreturnStmt *stmt) {
        VAST_UNREACHABLE("unsupported CoreturnStmt");
    }

    ValueOrStmt CodeGenVisitor::VisitCoroutineSuspendExpr(clang::CoroutineSuspendExpr *expr) {
        VAST_UNREACHABLE("unsupported CoroutineSuspendExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitCoawaitExpr(clang::CoawaitExpr *expr) {
        VAST_UNREACHABLE("unsupported CoawaitExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitCoyieldExpr(clang::CoyieldExpr *expr) {
        VAST_UNREACHABLE("unsupported CoyieldExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitDependentCoawaitExpr(clang::DependentCoawaitExpr *expr) {
        VAST_UNREACHABLE("unsupported DependentCoawaitExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitAttributedStmt(clang::AttributedStmt *stmt) {
        VAST_UNREACHABLE("unsupported AttributedStmt");
    }

    // Statements

    ValueOrStmt CodeGenVisitor::VisitBreakStmt(clang::BreakStmt *stmt) {
        auto loc = builder.get_location(stmt->getSourceRange());
        return builder.make< BreakOp >(loc);
    }

    ValueOrStmt CodeGenVisitor::VisitCXXCatchStmt(clang::CXXCatchStmt *stmt) {
        VAST_UNREACHABLE("unsupported CXXCatchStmt");
    }

    ValueOrStmt CodeGenVisitor::VisitCXXForRangeStmt(clang::CXXForRangeStmt *stmt) {
        VAST_UNREACHABLE("unsupported CXXForRangeStmt");
    }

    ValueOrStmt CodeGenVisitor::VisitCXXTryStmt(clang::CXXTryStmt *stmt) {
        VAST_UNREACHABLE("unsupported CXXTryStmt");
    }

    ValueOrStmt CodeGenVisitor::VisitCapturedStmt(clang::CapturedStmt *stmt) {
        VAST_UNREACHABLE("unsupported CapturedStmt");
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
        VAST_UNREACHABLE("unsupported AbstractConditionalOperator");
    }

    ValueOrStmt CodeGenVisitor::VisitBinaryConditionalOperator(
        clang::BinaryConditionalOperator *stmt) {
        VAST_UNREACHABLE("unsupported BinaryConditionalOperator");
    }

    ValueOrStmt CodeGenVisitor::VisitConditionalOperator(clang::ConditionalOperator *stmt) {
        VAST_UNREACHABLE("unsupported ConditionalOperator");
    }

    ValueOrStmt CodeGenVisitor::VisitAddrLabelExpr(clang::AddrLabelExpr *expr) {
        VAST_UNREACHABLE("unsupported AddrLabelExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitConstantExpr(clang::ConstantExpr *expr) {
        return CodeGenVisitor::Visit(expr->getSubExpr());
    }

    ValueOrStmt CodeGenVisitor::VisitArraySubscriptExpr(clang::ArraySubscriptExpr *expr) {
        auto loc    = builder.get_location(expr->getSourceRange());
        auto rty    = types.lvalue_convert(expr->getType());
        auto base   = CodeGenVisitor::Visit(expr->getBase());
        auto offset = CodeGenVisitor::Visit(expr->getIdx());
        return builder.make_value< SubscriptOp >(loc, rty, base, offset);
    }

    ValueOrStmt CodeGenVisitor::VisitArrayTypeTraitExpr(clang::ArrayTypeTraitExpr *expr) {
        VAST_UNREACHABLE("unsupported ArrayTypeTraitExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitAsTypeExpr(clang::AsTypeExpr *expr) {
        VAST_UNREACHABLE("unsupported AsTypeExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitAtomicExpr(clang::AtomicExpr *expr) {
        VAST_UNREACHABLE("unsupported AtomicExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitBlockExpr(clang::BlockExpr *expr) {
        VAST_UNREACHABLE("unsupported BlockExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitCXXBindTemporaryExpr(clang::CXXBindTemporaryExpr *expr) {
        VAST_UNREACHABLE("unsupported CXXBindTemporaryExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitCXXBoolLiteralExpr(const clang::CXXBoolLiteralExpr *lit) {
        return make_scalar_literal(lit);
    }

    ValueOrStmt CodeGenVisitor::VisitCXXConstructExpr(clang::CXXConstructExpr *expr) {
        VAST_UNREACHABLE("unsupported CXXConstructExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitCXXTemporaryObjectExpr(clang::CXXTemporaryObjectExpr *expr) {
        VAST_UNREACHABLE("unsupported CXXTemporaryObjectExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitCXXDefaultArgExpr(clang::CXXDefaultArgExpr *expr) {
        VAST_UNREACHABLE("unsupported CXXDefaultArgExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitCXXDefaultInitExpr(clang::CXXDefaultInitExpr *expr) {
        VAST_UNREACHABLE("unsupported CXXDefaultInitExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitCXXDeleteExpr(clang::CXXDeleteExpr *expr) {
        VAST_UNREACHABLE("unsupported CXXDeleteExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitCXXDependentScopeMemberExpr(
        clang::CXXDependentScopeMemberExpr *expr) {
        VAST_UNREACHABLE("unsupported CXXDependentScopeMemberExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitCXXNewExpr(clang::CXXNewExpr *expr) {
        VAST_UNREACHABLE("unsupported CXXNewExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitCXXNoexceptExpr(clang::CXXNoexceptExpr *expr) {
        VAST_UNREACHABLE("unsupported CXXNoexceptExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitCXXNullPtrLiteralExpr(clang::CXXNullPtrLiteralExpr *expr) {
        VAST_UNREACHABLE("unsupported CXXNullPtrLiteralExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitCXXPseudoDestructorExpr(clang::CXXPseudoDestructorExpr *expr) {
        VAST_UNREACHABLE("unsupported CXXPseudoDestructorExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitCXXScalarValueInitExpr(clang::CXXScalarValueInitExpr *expr) {
        VAST_UNREACHABLE("unsupported CXXScalarValueInitExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitCXXStdInitializerListExpr(
        clang::CXXStdInitializerListExpr *expr) {
        VAST_UNREACHABLE("unsupported CXXStdInitializerListExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitCXXThisExpr(clang::CXXThisExpr *expr) {
        VAST_UNREACHABLE("unsupported CXXThisExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitCXXThrowExpr(clang::CXXThrowExpr *expr) {
        VAST_UNREACHABLE("unsupported CXXThrowExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitCXXTypeidExpr(clang::CXXTypeidExpr *expr) {
        VAST_UNREACHABLE("unsupported CXXTypeidExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitCXXFoldExpr(clang::CXXFoldExpr *expr) {
        VAST_UNREACHABLE("unsupported CXXFoldExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitCXXUnresolvedConstructExpr(
        clang::CXXUnresolvedConstructExpr *expr) {
        VAST_UNREACHABLE("unsupported CXXUnresolvedConstructExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitCXXUuidofExpr(clang::CXXUuidofExpr *expr) {
        VAST_UNREACHABLE("unsupported CXXUuidofExpr");
    }

    mlir::FuncOp CodeGenVisitor::VisitDirectCallee(clang::FunctionDecl *callee) {
        auto name = callee->getName();
        if (auto fn = ctx.lookup_function(name, false /* with error */)) {
            return fn;
        }

        ScopedInsertPoint builder_scope(builder);
        builder.set_insertion_point_to_start(ctx.getModule()->getBody());
        auto stmt = std::get< Stmt >(VisitFunctionDecl(callee));
        return mlir::cast< mlir::FuncOp >(stmt);
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
        VAST_UNREACHABLE("unsupported CUDAKernelCallExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitCXXMemberCallExpr(clang::CXXMemberCallExpr *expr) {
        VAST_UNREACHABLE("unsupported CXXMemberCallExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitCXXOperatorCallExpr(clang::CXXOperatorCallExpr *expr) {
        VAST_UNREACHABLE("unsupported CXXOperatorCallExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitUserDefinedLiteral(clang::UserDefinedLiteral *lit) {
        VAST_UNREACHABLE("unsupported UserDefinedLiteral");
    }

    ValueOrStmt CodeGenVisitor::VisitCStyleCastExpr(clang::CStyleCastExpr *expr) {
        return make_cast< CStyleCastOp >(expr);
    }

    ValueOrStmt CodeGenVisitor::VisitCXXFunctionalCastExpr(clang::CXXFunctionalCastExpr *expr) {
        VAST_UNREACHABLE("unsupported CXXFunctionalCastExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitCXXConstCastExpr(clang::CXXConstCastExpr *expr) {
        VAST_UNREACHABLE("unsupported CXXConstCastExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitCXXDynamicCastExpr(clang::CXXDynamicCastExpr *expr) {
        VAST_UNREACHABLE("unsupported CXXDynamicCastExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitCXXReinterpretCastExpr(clang::CXXReinterpretCastExpr *expr) {
        VAST_UNREACHABLE("unsupported CXXReinterpretCastExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitCXXStaticCastExpr(clang::CXXStaticCastExpr *expr) {
        VAST_UNREACHABLE("unsupported CXXStaticCastExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitObjCBridgedCastExpr(clang::ObjCBridgedCastExpr *expr) {
        VAST_UNREACHABLE("unsupported ObjCBridgedCastExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitImplicitCastExpr(clang::ImplicitCastExpr *expr) {
        return make_cast< ImplicitCastOp >(expr);
    }

    ValueOrStmt CodeGenVisitor::VisitCharacterLiteral(clang::CharacterLiteral *lit) {
        return make_scalar_literal(lit);
    }

    ValueOrStmt CodeGenVisitor::VisitChooseExpr(clang::ChooseExpr *expr) {
        VAST_UNREACHABLE("unsupported ChooseExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitCompoundLiteralExpr(clang::CompoundLiteralExpr *expr) {
        VAST_UNREACHABLE("unsupported CompoundLiteralExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitConvertVectorExpr(clang::ConvertVectorExpr *expr) {
        VAST_UNREACHABLE("unsupported ConvertVectorExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitDeclRefExpr(clang::DeclRefExpr *expr) {
        auto loc = builder.get_location(expr->getSourceRange());
        auto underlying = expr->getDecl()->getUnderlyingDecl();

        // TODO(Heno): deal with function declaration

        if (auto enum_decl  = clang::dyn_cast< clang::EnumConstantDecl >(underlying)) {
            auto val = ctx.enum_constants.lookup(enum_decl->getName());
            VAST_ASSERT(val);
            auto rty = types.convert(expr->getType());
            return builder.make_value< EnumRefOp >(loc, rty, val.name());
        }

        if (auto var_decl = clang::dyn_cast< clang::VarDecl >(underlying)) {
            auto val = ctx.vars.lookup(var_decl);
            VAST_ASSERT(val);

            auto rty = types.lvalue_convert(expr->getType());

            if (var_decl->isFileVarDecl()) {
                auto var = val.getDefiningOp< VarDecl >();
                auto name = mlir::StringAttr::get(&ctx.getMLIRContext(), var.name());
                val = builder.make_value< GlobalRefOp >(loc, rty, name);
            }

            return builder.make_value< DeclRefOp >(loc, rty, val);
        }

        VAST_UNREACHABLE("unknown underlying declaration to be referenced");
    }

    ValueOrStmt CodeGenVisitor::VisitDependentScopeDeclRefExpr(
        clang::DependentScopeDeclRefExpr *expr) {
        VAST_UNREACHABLE("unsupported DependentScopeDeclRefExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitDesignatedInitExpr(clang::DesignatedInitExpr *expr) {
        VAST_UNREACHABLE("unsupported DesignatedInitExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitExprWithCleanups(clang::ExprWithCleanups *expr) {
        VAST_UNREACHABLE("unsupported ExprWithCleanups");
    }

    ValueOrStmt CodeGenVisitor::VisitExpressionTraitExpr(clang::ExpressionTraitExpr *expr) {
        VAST_UNREACHABLE("unsupported ExpressionTraitExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitExtVectorElementExpr(clang::ExtVectorElementExpr *expr) {
        VAST_UNREACHABLE("unsupported ExtVectorElementExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitFloatingLiteral(clang::FloatingLiteral *lit) {
        return make_scalar_literal(lit);
    }

    ValueOrStmt CodeGenVisitor::VisitFunctionParmPackExpr(clang::FunctionParmPackExpr *expr) {
        VAST_UNREACHABLE("unsupported FunctionParmPackExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitGNUNullExpr(clang::GNUNullExpr *expr) {
        VAST_UNREACHABLE("unsupported GNUNullExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitGenericSelectionExpr(clang::GenericSelectionExpr *expr) {
        VAST_UNREACHABLE("unsupported GenericSelectionExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitImaginaryLiteral(clang::ImaginaryLiteral *lit) {
        VAST_UNREACHABLE("unsupported ImaginaryLiteral");
    }

    ValueOrStmt CodeGenVisitor::VisitFixedPointLiteral(clang::FixedPointLiteral *lit) {
        VAST_UNREACHABLE("unsupported FixedPointLiteral");
    }

    ValueOrStmt CodeGenVisitor::VisitImplicitValueInitExpr(clang::ImplicitValueInitExpr *expr) {
        VAST_UNREACHABLE("unsupported ImplicitValueInitExpr");
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
        VAST_UNREACHABLE("unsupported LambdaExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitMSPropertyRefExpr(clang::MSPropertyRefExpr *expr) {
        VAST_UNREACHABLE("unsupported MSPropertyRefExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitMaterializeTemporaryExpr(
        clang::MaterializeTemporaryExpr *expr) {
        VAST_UNREACHABLE("unsupported MaterializeTemporaryExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitMemberExpr(clang::MemberExpr *expr) {
        auto loc   = builder.get_location(expr->getSourceRange());
        auto field = llvm::dyn_cast< clang::FieldDecl >(expr->getMemberDecl());
        auto name  = get_field_name(field);
        auto base  = CodeGenVisitor::Visit(expr->getBase());
        auto type  = types.lvalue_convert(expr->getType());
        return builder.make_value< RecordMemberOp >(loc, type, base, name);
    }

    ValueOrStmt CodeGenVisitor::VisitObjCArrayLiteral(clang::ObjCArrayLiteral *expr) {
        VAST_UNREACHABLE("unsupported ObjCArrayLiteral");
    }

    ValueOrStmt CodeGenVisitor::VisitObjCBoolLiteralExpr(clang::ObjCBoolLiteralExpr *expr) {
        VAST_UNREACHABLE("unsupported ObjCBoolLiteralExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitObjCBoxedExpr(clang::ObjCBoxedExpr *expr) {
        VAST_UNREACHABLE("unsupported ObjCBoxedExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitObjCDictionaryLiteral(clang::ObjCDictionaryLiteral *lit) {
        VAST_UNREACHABLE("unsupported ObjCDictionaryLiteral");
    }

    ValueOrStmt CodeGenVisitor::VisitObjCEncodeExpr(clang::ObjCEncodeExpr *expr) {
        VAST_UNREACHABLE("unsupported ObjCEncodeExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitObjCIndirectCopyRestoreExpr(
        clang::ObjCIndirectCopyRestoreExpr *expr) {
        VAST_UNREACHABLE("unsupported ObjCIndirectCopyRestoreExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitObjCIsaExpr(clang::ObjCIsaExpr *expr) {
        VAST_UNREACHABLE("unsupported ObjCIsaExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitObjCIvarRefExpr(clang::ObjCIvarRefExpr *expr) {
        VAST_UNREACHABLE("unsupported ObjCIvarRefExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitObjCMessageExpr(clang::ObjCMessageExpr *expr) {
        VAST_UNREACHABLE("unsupported ObjCMessageExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitObjCPropertyRefExpr(clang::ObjCPropertyRefExpr *expr) {
        VAST_UNREACHABLE("unsupported ObjCPropertyRefExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitObjCProtocolExpr(clang::ObjCProtocolExpr *expr) {
        VAST_UNREACHABLE("unsupported ObjCProtocolExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitObjCSelectorExpr(clang::ObjCSelectorExpr *expr) {
        VAST_UNREACHABLE("unsupported ObjCSelectorExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitObjCStringLiteral(clang::ObjCStringLiteral *lit) {
        VAST_UNREACHABLE("unsupported ObjCStringLiteral");
    }

    ValueOrStmt CodeGenVisitor::VisitObjCSubscriptRefExpr(clang::ObjCSubscriptRefExpr *expr) {
        VAST_UNREACHABLE("unsupported ObjCSubscriptRefExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitOffsetOfExpr(clang::OffsetOfExpr *expr) {
        VAST_UNREACHABLE("unsupported OffsetOfExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitOpaqueValueExpr(clang::OpaqueValueExpr *expr) {
        VAST_UNREACHABLE("unsupported OpaqueValueExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitOverloadExpr(clang::OverloadExpr *expr) {
        VAST_UNREACHABLE("unsupported OverloadExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitUnresolvedLookupExpr(clang::UnresolvedLookupExpr *expr) {
        VAST_UNREACHABLE("unsupported UnresolvedLookupExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitUnresolvedMemberExpr(clang::UnresolvedMemberExpr *expr) {
        VAST_UNREACHABLE("unsupported UnresolvedMemberExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitPackExpansionExpr(clang::PackExpansionExpr *expr) {
        VAST_UNREACHABLE("unsupported PackExpansionExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitParenExpr(clang::ParenExpr *expr) {
        auto loc     = builder.get_location(expr->getSourceRange());
        auto rty     = types.convert(expr->getType());
        auto subexpr = make_value_builder(expr->getSubExpr());
        return builder.make_value< ExprOp >(loc, rty, subexpr);
    }

    ValueOrStmt CodeGenVisitor::VisitParenListExpr(clang::ParenListExpr *expr) {
        VAST_UNREACHABLE("unsupported ParenListExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitPredefinedExpr(clang::PredefinedExpr *expr) {
        VAST_UNREACHABLE("unsupported PredefinedExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitPseudoObjectExpr(clang::PseudoObjectExpr *expr) {
        VAST_UNREACHABLE("unsupported PseudoObjectExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitShuffleVectorExpr(clang::ShuffleVectorExpr *expr) {
        VAST_UNREACHABLE("unsupported ShuffleVectorExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitSizeOfPackExpr(clang::SizeOfPackExpr *expr) {
        VAST_UNREACHABLE("unsupported SizeOfPackExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitStmtExpr(clang::StmtExpr *expr) {
        VAST_UNREACHABLE("unsupported StmtExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitStringLiteral(clang::StringLiteral *lit) {
        auto type = types.convert(lit->getType());
        auto loc  = builder.get_location(lit->getSourceRange());
        return builder.constant(loc, type, lit->getString());
    }

    ValueOrStmt CodeGenVisitor::VisitSubstNonTypeTemplateParmExpr(
        clang::SubstNonTypeTemplateParmExpr *expr) {
        VAST_UNREACHABLE("unsupported SubstNonTypeTemplateParmExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitSubstNonTypeTemplateParmPackExpr(
        clang::SubstNonTypeTemplateParmPackExpr *expr) {
        VAST_UNREACHABLE("unsupported SubstNonTypeTemplateParmPackExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitTypeTraitExpr(clang::TypeTraitExpr *expr) {
        VAST_UNREACHABLE("unsupported TypeTraitExpr");
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

        VAST_UNREACHABLE("unsupported UnaryExprOrTypeTraitExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitSourceLocExpr(clang::SourceLocExpr *expr) {
        VAST_UNREACHABLE("unsupported SourceLocExpr");
    }

    ValueOrStmt CodeGenVisitor::VisitVAArgExpr(clang::VAArgExpr *expr) {
        VAST_UNREACHABLE("unsupported VAArgExpr");
    }

    // Statements

    ValueOrStmt CodeGenVisitor::VisitForStmt(clang::ForStmt *stmt) {
        auto loc = builder.get_location(stmt->getSourceRange());

        auto make_loop_op = [&] {
            auto incr = make_region_builder(stmt->getInc());
            auto body = make_region_builder(stmt->getBody());
            if (auto cond = stmt->getCond())
                return builder.make< ForOp >(loc, make_cond_builder(cond), incr, body);
            return builder.make< ForOp >(loc, make_yield_true(), incr, body);
        };

        if (stmt->getInit()) {
            return make_scoped(builder, loc, [&] {
                Visit(stmt->getInit());
                make_loop_op();
            });
        }

        return make_loop_op();
    }

    ValueOrStmt CodeGenVisitor::VisitGotoStmt(clang::GotoStmt *stmt) {
        VAST_UNREACHABLE("unsupported GotoStmt");
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
        VAST_UNREACHABLE("unsupported IndirectGotoStmt");
    }

    ValueOrStmt CodeGenVisitor::VisitLabelStmt(clang::LabelStmt *stmt) {
        VAST_UNREACHABLE("unsupported LabelStmt");
    }

    ValueOrStmt CodeGenVisitor::VisitMSDependentExistsStmt(clang::MSDependentExistsStmt *stmt) {
        VAST_UNREACHABLE("unsupported MSDependentExistsStmt");
    }

    ValueOrStmt CodeGenVisitor::VisitNullStmt(clang::NullStmt *stmt) {
        VAST_UNREACHABLE("unsupported NullStmt");
    }

    ValueOrStmt CodeGenVisitor::VisitOMPBarrierDirective(clang::OMPBarrierDirective *dir) {
        VAST_UNREACHABLE("unsupported OMPBarrierDirective");
    }

    ValueOrStmt CodeGenVisitor::VisitOMPCriticalDirective(clang::OMPCriticalDirective *dir) {
        VAST_UNREACHABLE("unsupported OMPCriticalDirective");
    }

    ValueOrStmt CodeGenVisitor::VisitOMPFlushDirective(clang::OMPFlushDirective *dir) {
        VAST_UNREACHABLE("unsupported OMPFlushDirective");
    }

    ValueOrStmt CodeGenVisitor::VisitOMPForDirective(clang::OMPForDirective *dir) {
        VAST_UNREACHABLE("unsupported OMPForDirective");
    }

    ValueOrStmt CodeGenVisitor::VisitOMPMasterDirective(clang::OMPMasterDirective *dir) {
        VAST_UNREACHABLE("unsupported OMPMasterDirective");
    }

    ValueOrStmt CodeGenVisitor::VisitOMPParallelDirective(clang::OMPParallelDirective *dir) {
        VAST_UNREACHABLE("unsupported OMPParallelDirective");
    }

    ValueOrStmt CodeGenVisitor::VisitOMPParallelForDirective(clang::OMPParallelForDirective *dir) {
        VAST_UNREACHABLE("unsupported OMPParallelForDirective");
    }

    ValueOrStmt CodeGenVisitor::VisitOMPParallelSectionsDirective(
        clang::OMPParallelSectionsDirective *dir) {
        VAST_UNREACHABLE("unsupported OMPParallelSectionsDirective");
    }

    ValueOrStmt CodeGenVisitor::VisitOMPSectionDirective(clang::OMPSectionDirective *dir) {
        VAST_UNREACHABLE("unsupported OMPSectionDirective");
    }

    ValueOrStmt CodeGenVisitor::VisitOMPSectionsDirective(clang::OMPSectionsDirective *dir) {
        VAST_UNREACHABLE("unsupported OMPSectionsDirective");
    }

    ValueOrStmt CodeGenVisitor::VisitOMPSimdDirective(clang::OMPSimdDirective *dir) {
        VAST_UNREACHABLE("unsupported OMPSimdDirective");
    }

    ValueOrStmt CodeGenVisitor::VisitOMPSingleDirective(clang::OMPSingleDirective *dir) {
        VAST_UNREACHABLE("unsupported OMPSingleDirective");
    }

    ValueOrStmt CodeGenVisitor::VisitOMPTaskDirective(clang::OMPTaskDirective *dir) {
        VAST_UNREACHABLE("unsupported OMPTaskDirective");
    }

    ValueOrStmt CodeGenVisitor::VisitOMPTaskwaitDirective(clang::OMPTaskwaitDirective *dir) {
        VAST_UNREACHABLE("unsupported OMPTaskwaitDirective");
    }

    ValueOrStmt CodeGenVisitor::VisitOMPTaskyieldDirective(clang::OMPTaskyieldDirective *dir) {
        VAST_UNREACHABLE("unsupported OMPTaskyieldDirective");
    }

    ValueOrStmt CodeGenVisitor::VisitObjCAtCatchStmt(clang::ObjCAtCatchStmt *stmt) {
        VAST_UNREACHABLE("unsupported ObjCAtCatchStmt");
    }

    ValueOrStmt CodeGenVisitor::VisitObjCAtFinallyStmt(clang::ObjCAtFinallyStmt *stmt) {
        VAST_UNREACHABLE("unsupported ObjCAtFinallyStmt");
    }

    ValueOrStmt CodeGenVisitor::VisitObjCAtSynchronizedStmt(clang::ObjCAtSynchronizedStmt *stmt) {
        VAST_UNREACHABLE("unsupported ObjCAtSynchronizedStmt");
    }

    ValueOrStmt CodeGenVisitor::VisitObjCAtThrowStmt(clang::ObjCAtThrowStmt *stmt) {
        VAST_UNREACHABLE("unsupported ObjCAtThrowStmt");
    }

    ValueOrStmt CodeGenVisitor::VisitObjCAtTryStmt(clang::ObjCAtTryStmt *stmt) {
        VAST_UNREACHABLE("unsupported ObjCAtTryStmt");
    }

    ValueOrStmt CodeGenVisitor::VisitObjCAutoreleasePoolStmt(clang::ObjCAutoreleasePoolStmt *stmt) {
        VAST_UNREACHABLE("unsupported ObjCAutoreleasePoolStmt");
    }

    ValueOrStmt CodeGenVisitor::VisitObjCForCollectionStmt(clang::ObjCForCollectionStmt *stmt) {
        VAST_UNREACHABLE("unsupported ObjCForCollectionStmt");
    }

    ValueOrStmt CodeGenVisitor::VisitReturnStmt(clang::ReturnStmt *stmt) {
        auto loc = builder.get_location(stmt->getSourceRange());
        if (auto ret = stmt->getRetValue())
            return builder.make< ReturnOp >(loc, CodeGenVisitor::Visit(ret));
        return builder.make< ReturnOp >(loc);
    }

    ValueOrStmt CodeGenVisitor::VisitSEHExceptStmt(clang::SEHExceptStmt *stmt) {
        VAST_UNREACHABLE("unsupported SEHExceptStmt");
    }

    ValueOrStmt CodeGenVisitor::VisitSEHFinallyStmt(clang::SEHFinallyStmt *stmt) {
        VAST_UNREACHABLE("unsupported SEHFinallyStmt");
    }

    ValueOrStmt CodeGenVisitor::VisitSEHLeaveStmt(clang::SEHLeaveStmt *stmt) {
        VAST_UNREACHABLE("unsupported SEHLeaveStmt");
    }

    ValueOrStmt CodeGenVisitor::VisitSEHTryStmt(clang::SEHTryStmt *stmt) {
        VAST_UNREACHABLE("unsupported SEHTryStmt");
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
        auto loc = builder.get_location(stmt->getSourceRange());

        auto make_switch_op = [&] {
            auto cond_builder = make_value_builder(stmt->getCond());
            auto body_builder = make_region_builder(stmt->getBody());
            return builder.make< SwitchOp >(loc, cond_builder, body_builder);
        };

        if (stmt->getInit()) {
            return make_scoped(builder, loc, [&] {
                Visit(stmt->getInit());
                make_switch_op();
            });
        }

        return make_switch_op();
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
        VAST_UNREACHABLE("unsupported ImportDecl");
    }

    ValueOrStmt CodeGenVisitor::VisitEmptyDecl(clang::EmptyDecl *decl) {
        VAST_UNREACHABLE("unsupported EmptyDecl");
    }

    ValueOrStmt CodeGenVisitor::VisitAccessSpecDecl(clang::AccessSpecDecl *decl) {
        VAST_UNREACHABLE("unsupported AccessSpecDecl");
    }

    ValueOrStmt CodeGenVisitor::VisitCapturedDecl(clang::CapturedDecl *decl) {
        VAST_UNREACHABLE("unsupported CapturedDecl");
    }

    ValueOrStmt CodeGenVisitor::VisitClassScopeFunctionSpecializationDecl(
        clang::ClassScopeFunctionSpecializationDecl *decl) {
        VAST_UNREACHABLE("unsupported ClassScopeFunctionSpecializationDecl");
    }

    ValueOrStmt CodeGenVisitor::VisitExportDecl(clang::ExportDecl *decl) {
        VAST_UNREACHABLE("unsupported ExportDecl");
    }

    ValueOrStmt CodeGenVisitor::VisitExternCContextDecl(clang::ExternCContextDecl *decl) {
        VAST_UNREACHABLE("unsupported ExternCContextDecl");
    }

    ValueOrStmt CodeGenVisitor::VisitFileScopeAsmDecl(clang::FileScopeAsmDecl *decl) {
        VAST_UNREACHABLE("unsupported FileScopeAsmDecl");
    }

    ValueOrStmt CodeGenVisitor::VisitStaticAssertDecl(clang::StaticAssertDecl *decl) {
        VAST_UNREACHABLE("unsupported StaticAssertDecl");
    }

    ValueOrStmt CodeGenVisitor::VisitTranslationUnitDecl(clang::TranslationUnitDecl *tu) {
        ScopedInsertPoint builder_scope(builder);

        auto loc = builder.get_location(tu->getSourceRange());

        auto unit  = builder.make< TranslationUnitOp >(loc);
        // TODO(Heno): refactor out together with from source codegen
        unit.body().push_back(new mlir::Block());
        builder.set_insertion_point_to_start(&unit.body().front());

        for (const auto &decl : tu->decls()) {
            CodeGenVisitor::Visit(decl);
        }

        // parform after we gather all types from the translation unit
        emit_data_layout(ctx.getMLIRContext(), ctx.getModule(), ctx.data_layout());
        return unit;
    }

    ValueOrStmt CodeGenVisitor::VisitBindingDecl(clang::BindingDecl *decl) {
        VAST_UNREACHABLE("unsupported BindingDecl");
    }

    // ValueOrStmt CodeGenVisitor::VisitNamespaceDecl(clang::NamespaceDecl *decl)
    // {
    //     VAST_UNREACHABLE( "unsupported NamespaceDecl" );
    // }

    ValueOrStmt CodeGenVisitor::VisitNamespaceAliasDecl(clang::NamespaceAliasDecl *decl) {
        VAST_UNREACHABLE("unsupported NamespaceAliasDecl");
    }

    // ValueOrStmt CodeGenVisitor::VisitTypedefNameDecl(clang::TypedefNameDecl *decl)
    // {
    //     VAST_UNREACHABLE( "unsupported TypedefNameDecl" );
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

        auto def = builder.define_type(loc, type, name);
        attach_attributes(decl /* from */, def /* to */);
        return def;
    }

    ValueOrStmt CodeGenVisitor::VisitTypeAliasDecl(clang::TypeAliasDecl *decl) {
        VAST_UNREACHABLE("unsupported TypeAliasDecl");
    }
    ValueOrStmt CodeGenVisitor::VisitTemplateDecl(clang::TemplateDecl *decl) {
        VAST_UNREACHABLE("unsupported TemplateDecl");
    }

    ValueOrStmt CodeGenVisitor::VisitTypeAliasTemplateDecl(clang::TypeAliasTemplateDecl *decl) {
        VAST_UNREACHABLE("unsupported TypeAliasTemplateDecl");
    }

    ValueOrStmt CodeGenVisitor::VisitLabelDecl(clang::LabelDecl *decl) {
        VAST_UNREACHABLE("unsupported LabelDecl");
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
        if (decl->isUnion()) {
            return make_record_decl< UnionDeclOp >(decl);
        } else {
            return make_record_decl< RecordDeclOp >(decl);
        }
    }

    ValueOrStmt CodeGenVisitor::VisitEnumConstantDecl(clang::EnumConstantDecl *decl) {
        auto initializer = make_value_builder(decl->getInitExpr());

        return builder.declare_enum_constant(make_operation< EnumConstantOp >(builder)
            .bind(builder.get_end_location(decl->getSourceRange())) // location
            .bind(decl->getName())                                  // name
            .bind(decl->getInitVal())                               // value
            .bind_if(decl->getInitExpr(), std::move(initializer))   // initializer
            .freeze()
        );
    }

    ValueOrStmt CodeGenVisitor::VisitFunctionDecl(clang::FunctionDecl *decl) {
        auto name = decl->getName();
        auto is_definition = decl->doesThisDeclarationHaveABody();

        // emit definition instead of declaration
        if (!is_definition && decl->getDefinition()) {
            return Visit(decl->getDefinition());
        }

        // return already seen definition
        if (auto fn = ctx.functions.lookup(name)) {
            return fn;
        }

        ScopedInsertPoint builder_scope(builder);
        llvm::ScopedHashTableScope scope(ctx.vars);

        auto loc  = builder.get_location(decl->getSourceRange());
        auto type = types.convert(decl->getFunctionType());
        VAST_ASSERT(type);

        // create function header, that will be later filled with function body
        // or returned as declaration in the case of external function
        auto fn = builder.make< mlir::FuncOp >(loc, name, type);
        if (failed(ctx.functions.declare(name, fn))) {
            ctx.error("error: multiple declarations of a same function" + name);
        }

        if (!is_definition) {
            fn.setVisibility( mlir::FuncOp::Visibility::Private );
            return fn;
        }

        // emit function body
        auto entry = fn.addEntryBlock();
        builder.set_insertion_point_to_start(entry);

        if (decl->hasBody()) {
            // In MLIR the entry block of the function must have the same
            // argument list as the function itself.
            auto params = llvm::zip(decl->getDefinition()->parameters(), entry->getArguments());
            for (const auto &[arg, earg] : params) {
                if (failed(ctx.vars.declare(arg, earg)))
                    ctx.error("error: multiple declarations of a same symbol" + arg->getName());
            }

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
        VAST_UNREACHABLE("unsupported CXXMethodDecl");
    }

    ValueOrStmt CodeGenVisitor::VisitCXXConstructorDecl(clang::CXXConstructorDecl *decl) {
        VAST_UNREACHABLE("unsupported CXXConstructorDecl");
    }

    ValueOrStmt CodeGenVisitor::VisitCXXDestructorDecl(clang::CXXDestructorDecl *decl) {
        VAST_UNREACHABLE("unsupported CXXDestructorDecl");
    }

    ValueOrStmt CodeGenVisitor::VisitCXXConversionDecl(clang::CXXConversionDecl *decl) {
        VAST_UNREACHABLE("unsupported CXXConversionDecl");
    }

    ValueOrStmt CodeGenVisitor::VisitCXXDeductionGuideDecl(clang::CXXDeductionGuideDecl *decl) {
        VAST_UNREACHABLE("unsupported CXXDeductionGuideDecl");
    }

    ValueOrStmt CodeGenVisitor::VisitMSPropertyDecl(clang::MSPropertyDecl *decl) {
        VAST_UNREACHABLE("unsupported MSPropertyDecl");
    }

    ValueOrStmt CodeGenVisitor::VisitMSGuidDecl(clang::MSGuidDecl *decl) {
        VAST_UNREACHABLE("unsupported MSGuidDecl");
    }

    ValueOrStmt CodeGenVisitor::VisitFieldDecl(clang::FieldDecl *decl) {
        auto loc  = builder.get_location(decl->getSourceRange());
        auto name = get_field_name(decl);

        // define field type if the field defines a new nested type
        if (auto tag = decl->getType()->getAsTagDecl()) {
            if (tag->isThisDeclarationADefinition()) {
                if (!ctx.tag_names.count(tag)) {
                    Visit(tag);
                }
            }
        }
        auto type = types.convert(decl->getType());
        return builder.make< FieldDeclOp >(loc, name, type);
    }

    ValueOrStmt CodeGenVisitor::VisitIndirectFieldDecl(clang::IndirectFieldDecl *decl) {
        VAST_UNREACHABLE("unsupported IndirectFieldDecl");
    }

    ValueOrStmt CodeGenVisitor::VisitFriendDecl(clang::FriendDecl *decl) {
        VAST_UNREACHABLE("unsupported FriendDecl");
    }

    ValueOrStmt CodeGenVisitor::VisitFriendTemplateDecl(clang::FriendTemplateDecl *decl) {
        VAST_UNREACHABLE("unsupported FriendTemplateDecl");
    }

    ValueOrStmt CodeGenVisitor::VisitObjCAtDefsFieldDecl(clang::ObjCAtDefsFieldDecl *decl) {
        VAST_UNREACHABLE("unsupported ObjCAtDefsFieldDecl");
    }

    ValueOrStmt CodeGenVisitor::VisitObjCIvarDecl(clang::ObjCIvarDecl *decl) {
        VAST_UNREACHABLE("unsupported ObjCIvarDecl");
    }

    StorageClass get_storage_class(clang::VarDecl *decl) {
        switch (decl->getStorageClass()) {
            case clang::SC_None: return StorageClass::sc_none;
            case clang::SC_Auto: return StorageClass::sc_auto;
            case clang::SC_Static: return StorageClass::sc_static;
            case clang::SC_Extern: return StorageClass::sc_extern;
            case clang::SC_PrivateExtern: return StorageClass::sc_private_extern;
            case clang::SC_Register: return StorageClass::sc_register;
        }
    }

    TSClass get_thread_storage_class(clang::VarDecl *decl) {
        switch (decl->getTSCSpec()) {
            case clang::TSCS_unspecified: return TSClass::tsc_none;
            case clang::TSCS___thread: return TSClass::tsc_gnu_thread;
            case clang::TSCS_thread_local: return TSClass::tsc_cxx_thread;
            case clang::TSCS__Thread_local: return TSClass::tsc_c_thread;
        }
    }

    ValueOrStmt CodeGenVisitor::VisitVarDecl(clang::VarDecl *decl) {
        auto initializer = make_value_builder(decl->getInit());

        auto array_allocator = [decl, this](auto &bld, auto loc) {
            if (auto type = clang::dyn_cast< clang::VariableArrayType >(decl->getType())) {
                make_value_builder(type->getSizeExpr())(bld, loc);
            }
        };

        auto type = decl->getType();
        bool has_allocator = type->isVariableArrayType();
        bool has_init =  decl->getInit();

        auto var = make_operation< VarDecl >(builder)
            .bind(builder.get_end_location(decl->getSourceRange())) // location
            .bind(types.lvalue_convert(decl->getType()))            // type
            .bind(decl->getUnderlyingDecl()->getName())             // name
            .bind_region_if(has_init, std::move(initializer))           // initializer
            .bind_region_if(has_allocator, std::move(array_allocator))  // array allocator
            .freeze();

        if (auto sc = get_storage_class(decl); sc != StorageClass::sc_none) {
            var.setStorageClass(sc);
        }

        if (auto tsc = get_thread_storage_class(decl); tsc != TSClass::tsc_none) {
            var.setThreadStorageClass(tsc);
        }

        if (failed(ctx.vars.declare(decl, var))) {
            ctx.error("error: multiple declarations of a same symbol" + decl->getName());
        }

        return mlir::Value(var);
    }

    ValueOrStmt CodeGenVisitor::VisitDecompositionDecl(clang::DecompositionDecl *decl) {
        VAST_UNREACHABLE("unsupported DecompositionDecl");
    }

    ValueOrStmt CodeGenVisitor::VisitImplicitParamDecl(clang::ImplicitParamDecl *decl) {
        VAST_UNREACHABLE("unsupported ImplicitParamDecl");
    }

    // ValueOrStmt
    // CodeGenVisitor::VisitUnresolvedUsingIfExistsDecl(clang::UnresolvedUsingIfExistsDecl
    // *decl)
    // {
    //     VAST_UNREACHABLE( "unsupported UnresolvedUsingIfExistsDecl" );
    // }

    ValueOrStmt CodeGenVisitor::VisitParmVarDecl(clang::ParmVarDecl *decl) {
        if (auto var = ctx.vars.lookup(decl))
            return var;
        VAST_UNREACHABLE("Missing parameter declaration {}", decl->getName());
    }

    ValueOrStmt CodeGenVisitor::VisitObjCMethodDecl(clang::ObjCMethodDecl *decl) {
        VAST_UNREACHABLE("unsupported ObjCMethodDecl");
    }

    ValueOrStmt CodeGenVisitor::VisitObjCTypeParamDecl(clang::ObjCTypeParamDecl *decl) {
        VAST_UNREACHABLE("unsupported ObjCTypeParamDecl");
    }

    ValueOrStmt CodeGenVisitor::VisitObjCProtocolDecl(clang::ObjCProtocolDecl *decl) {
        VAST_UNREACHABLE("unsupported ObjCProtocolDecl");
    }

    ValueOrStmt CodeGenVisitor::VisitLinkageSpecDecl(clang::LinkageSpecDecl *decl) {
        VAST_UNREACHABLE("unsupported LinkageSpecDecl");
    }

    ValueOrStmt CodeGenVisitor::VisitUsingDecl(clang::UsingDecl *decl) {
        VAST_UNREACHABLE("unsupported UsingDecl");
    }

    ValueOrStmt CodeGenVisitor::VisitUsingShadowDecl(clang::UsingShadowDecl *decl) {
        VAST_UNREACHABLE("unsupported UsingShadowDecl");
    }

    ValueOrStmt CodeGenVisitor::VisitUsingDirectiveDecl(clang::UsingDirectiveDecl *decl) {
        VAST_UNREACHABLE("unsupported UsingDirectiveDecl");
    }

    ValueOrStmt CodeGenVisitor::VisitUsingPackDecl(clang::UsingPackDecl *decl) {
        VAST_UNREACHABLE("unsupported UsingPackDecl");
    }

    // ValueOrStmt CodeGenVisitor::VisitUsingEnumDecl(clang::UsingEnumDecl *decl)
    // {
    //     VAST_UNREACHABLE( "unsupported UsingEnumDecl" );
    // }

    ValueOrStmt CodeGenVisitor::VisitUnresolvedUsingValueDecl(
        clang::UnresolvedUsingValueDecl *decl) {
        VAST_UNREACHABLE("unsupported UnresolvedUsingValueDecl");
    }

    ValueOrStmt CodeGenVisitor::VisitUnresolvedUsingTypenameDecl(
        clang::UnresolvedUsingTypenameDecl *decl) {
        VAST_UNREACHABLE("unsupported UnresolvedUsingTypenameDecl");
    }

    ValueOrStmt CodeGenVisitor::VisitBuiltinTemplateDecl(clang::BuiltinTemplateDecl *decl) {
        VAST_UNREACHABLE("unsupported BuiltinTemplateDecl");
    }

    ValueOrStmt CodeGenVisitor::VisitConceptDecl(clang::ConceptDecl *decl) {
        VAST_UNREACHABLE("unsupported ConceptDecl");
    }

    ValueOrStmt CodeGenVisitor::VisitRedeclarableTemplateDecl(
        clang::RedeclarableTemplateDecl *decl) {
        VAST_UNREACHABLE("unsupported RedeclarableTemplateDecl");
    }

    ValueOrStmt CodeGenVisitor::VisitLifetimeExtendedTemporaryDecl(
        clang::LifetimeExtendedTemporaryDecl *decl) {
        VAST_UNREACHABLE("unsupported LifetimeExtendedTemporaryDecl");
    }

    ValueOrStmt CodeGenVisitor::VisitPragmaCommentDecl(clang::PragmaCommentDecl *decl) {
        VAST_UNREACHABLE("unsupported PragmaCommentDecl");
    }

    ValueOrStmt CodeGenVisitor::VisitPragmaDetectMismatchDecl(
        clang::PragmaDetectMismatchDecl *decl) {
        VAST_UNREACHABLE("unsupported PragmaDetectMismatchDecl");
    }

    ValueOrStmt CodeGenVisitor::VisitRequiresExprBodyDecl(clang::RequiresExprBodyDecl *decl) {
        VAST_UNREACHABLE("unsupported RequiresExprBodyDecl");
    }

    ValueOrStmt CodeGenVisitor::VisitObjCCompatibleAliasDecl(clang::ObjCCompatibleAliasDecl *decl) {
        VAST_UNREACHABLE("unsupported ObjCCompatibleAliasDecl");
    }

    ValueOrStmt CodeGenVisitor::VisitObjCCategoryDecl(clang::ObjCCategoryDecl *decl) {
        VAST_UNREACHABLE("unsupported ObjCCategoryDecl");
    }

    ValueOrStmt CodeGenVisitor::VisitObjCImplDecl(clang::ObjCImplDecl *decl) {
        VAST_UNREACHABLE("unsupported ObjCImplDecl");
    }

    ValueOrStmt CodeGenVisitor::VisitObjCInterfaceDecl(clang::ObjCInterfaceDecl *decl) {
        VAST_UNREACHABLE("unsupported ObjCInterfaceDecl");
    }

    ValueOrStmt CodeGenVisitor::VisitObjCCategoryImplDecl(clang::ObjCCategoryImplDecl *decl) {
        VAST_UNREACHABLE("unsupported ObjCCategoryImplDecl");
    }

    ValueOrStmt CodeGenVisitor::VisitObjCImplementationDecl(clang::ObjCImplementationDecl *decl) {
        VAST_UNREACHABLE("unsupported ObjCImplementationDecl");
    }

    ValueOrStmt CodeGenVisitor::VisitObjCPropertyDecl(clang::ObjCPropertyDecl *decl) {
        VAST_UNREACHABLE("unsupported ObjCPropertyDecl");
    }

    ValueOrStmt CodeGenVisitor::VisitObjCPropertyImplDecl(clang::ObjCPropertyImplDecl *decl) {
        VAST_UNREACHABLE("unsupported ObjCPropertyImplDecl");
    }

    ValueOrStmt CodeGenVisitor::VisitTemplateParamObjectDecl(clang::TemplateParamObjectDecl *decl) {
        VAST_UNREACHABLE("unsupported TemplateParamObjectDecl");
    }

    ValueOrStmt CodeGenVisitor::VisitTemplateTypeParmDecl(clang::TemplateTypeParmDecl *decl) {
        VAST_UNREACHABLE("unsupported TemplateTypeParmDecl");
    }

    ValueOrStmt CodeGenVisitor::VisitNonTypeTemplateParmDecl(clang::NonTypeTemplateParmDecl *decl) {
        VAST_UNREACHABLE("unsupported NonTypeTemplateParmDecl");
    }

    ValueOrStmt CodeGenVisitor::VisitTemplateTemplateParmDecl(
        clang::TemplateTemplateParmDecl *decl) {
        VAST_UNREACHABLE("unsupported TemplateTemplateParmDecl");
    }

    ValueOrStmt CodeGenVisitor::VisitClassTemplateDecl(clang::ClassTemplateDecl *decl) {
        VAST_UNREACHABLE("unsupported ClassTemplateDecl");
    }

    ValueOrStmt CodeGenVisitor::VisitClassTemplatePartialSpecializationDecl(
        clang::ClassTemplatePartialSpecializationDecl *decl) {
        VAST_UNREACHABLE("unsupported ClassTemplatePartialSpecializationDecl");
    }

    ValueOrStmt CodeGenVisitor::VisitClassTemplateSpecializationDecl(
        clang::ClassTemplateSpecializationDecl *decl) {
        VAST_UNREACHABLE("unsupported ClassTemplateSpecializationDecl");
    }

    ValueOrStmt CodeGenVisitor::VisitVarTemplateDecl(clang::VarTemplateDecl *decl) {
        VAST_UNREACHABLE("unsupported VarTemplateDecl");
    }

    ValueOrStmt CodeGenVisitor::VisitVarTemplateSpecializationDecl(
        clang::VarTemplateSpecializationDecl *decl) {
        VAST_UNREACHABLE("unsupported VarTemplateSpecializationDecl");
    }

    ValueOrStmt CodeGenVisitor::VisitVarTemplatePartialSpecializationDecl(
        clang::VarTemplatePartialSpecializationDecl *decl) {
        VAST_UNREACHABLE("unsupported VarTemplatePartialSpecializationDecl");
    }

    ValueOrStmt CodeGenVisitor::VisitFunctionTemplateDecl(clang::FunctionTemplateDecl *decl) {
        VAST_UNREACHABLE("unsupported FunctionTemplateDecl");
    }

    ValueOrStmt CodeGenVisitor::VisitConstructorUsingShadowDecl(
        clang::ConstructorUsingShadowDecl *decl) {
        VAST_UNREACHABLE("unsupported ConstructorUsingShadowDecl");
    }

    ValueOrStmt CodeGenVisitor::VisitOMPAllocateDecl(clang::OMPAllocateDecl *decl) {
        VAST_UNREACHABLE("unsupported OMPAllocateDecl");
    }

    ValueOrStmt CodeGenVisitor::VisitOMPRequiresDecl(clang::OMPRequiresDecl *decl) {
        VAST_UNREACHABLE("unsupported OMPRequiresDecl");
    }

    ValueOrStmt CodeGenVisitor::VisitOMPThreadPrivateDecl(clang::OMPThreadPrivateDecl *decl) {
        VAST_UNREACHABLE("unsupported OMPThreadPrivateDecl");
    }

    ValueOrStmt CodeGenVisitor::VisitOMPCapturedExprDecl(clang::OMPCapturedExprDecl *decl) {
        VAST_UNREACHABLE("unsupported OMPCapturedExprDecl");
    }

    ValueOrStmt CodeGenVisitor::VisitOMPDeclareReductionDecl(clang::OMPDeclareReductionDecl *decl) {
        VAST_UNREACHABLE("unsupported OMPDeclareReductionDecl");
    }

    ValueOrStmt CodeGenVisitor::VisitOMPDeclareMapperDecl(clang::OMPDeclareMapperDecl *decl) {
        VAST_UNREACHABLE("unsupported OMPDeclareMapperDecl");
    }

} // namespace vast::hl
