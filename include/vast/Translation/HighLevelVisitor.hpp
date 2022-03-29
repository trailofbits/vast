// Copyright (c) 2022-present, Trail of Bits, Inc.

#pragma once

#include "vast/Util/Warnings.hpp"

VAST_RELAX_WARNINGS
#include <clang/AST/DeclVisitor.h>
#include <clang/AST/Expr.h>
#include <clang/AST/RecursiveASTVisitor.h>
#include <clang/AST/Stmt.h>
#include <clang/AST/StmtVisitor.h>
VAST_UNRELAX_WARNINGS

#include "vast/Translation/Context.hpp"
#include "vast/Translation/Expr.hpp"
#include "vast/Translation/HighLevelBuilder.hpp"
#include "vast/Translation/HighLevelTypeConverter.hpp"
#include "vast/Translation/Util.hpp"
#include "vast/Util/DataLayout.hpp"
#include "vast/Util/Types.hpp"

namespace vast::hl
{
    struct VastDeclVisitor;

    struct CodeGenVisitor
        : clang::StmtVisitor< CodeGenVisitor, ValueOrStmt >
        , clang::DeclVisitor< CodeGenVisitor, ValueOrStmt >
    {
        CodeGenVisitor(TranslationContext &ctx)
            : ctx(ctx), builder(ctx), types(ctx)
        {}

        using StmtVisitor = clang::StmtVisitor< CodeGenVisitor, ValueOrStmt >;
        using DeclVisitor = clang::DeclVisitor< CodeGenVisitor, ValueOrStmt >;

        using DeclVisitor::Visit;
        using StmtVisitor::Visit;

        // Binary Operations

        ValueOrStmt VisitBinPtrMemD(clang::BinaryOperator *expr);
        ValueOrStmt VisitBinPtrMemI(clang::BinaryOperator *expr);
        ValueOrStmt VisitBinMul(clang::BinaryOperator *expr);
        ValueOrStmt VisitBinDiv(clang::BinaryOperator *expr);
        ValueOrStmt VisitBinRem(clang::BinaryOperator *expr);
        ValueOrStmt VisitBinAdd(clang::BinaryOperator *expr);
        ValueOrStmt VisitBinSub(clang::BinaryOperator *expr);
        ValueOrStmt VisitBinShl(clang::BinaryOperator *expr);
        ValueOrStmt VisitBinShr(clang::BinaryOperator *expr);
        ValueOrStmt VisitBinLT(clang::BinaryOperator *expr);
        ValueOrStmt VisitBinGT(clang::BinaryOperator *expr);
        ValueOrStmt VisitBinLE(clang::BinaryOperator *expr);
        ValueOrStmt VisitBinGE(clang::BinaryOperator *expr);
        ValueOrStmt VisitBinEQ(clang::BinaryOperator *expr);
        ValueOrStmt VisitBinNE(clang::BinaryOperator *expr);
        ValueOrStmt VisitBinAnd(clang::BinaryOperator *expr);
        ValueOrStmt VisitBinXor(clang::BinaryOperator *expr);
        ValueOrStmt VisitBinOr(clang::BinaryOperator *expr);
        ValueOrStmt VisitBinLAnd(clang::BinaryOperator *expr);
        ValueOrStmt VisitBinLOr(clang::BinaryOperator *expr);
        ValueOrStmt VisitBinAssign(clang::BinaryOperator *expr);

        // Compound Assignment Operations

        ValueOrStmt VisitBinMulAssign(clang::CompoundAssignOperator *expr);
        ValueOrStmt VisitBinDivAssign(clang::CompoundAssignOperator *expr);
        ValueOrStmt VisitBinRemAssign(clang::CompoundAssignOperator *expr);
        ValueOrStmt VisitBinAddAssign(clang::CompoundAssignOperator *expr);
        ValueOrStmt VisitBinSubAssign(clang::CompoundAssignOperator *expr);
        ValueOrStmt VisitBinShlAssign(clang::CompoundAssignOperator *expr);
        ValueOrStmt VisitBinShrAssign(clang::CompoundAssignOperator *expr);
        ValueOrStmt VisitBinAndAssign(clang::CompoundAssignOperator *expr);
        ValueOrStmt VisitBinOrAssign(clang::CompoundAssignOperator *expr);
        ValueOrStmt VisitBinXorAssign(clang::CompoundAssignOperator *expr);
        ValueOrStmt VisitBinComma(clang::BinaryOperator *expr);

        // Unary Operations

        ValueOrStmt VisitUnaryPostInc(clang::UnaryOperator *expr);
        ValueOrStmt VisitUnaryPostDec(clang::UnaryOperator *expr);
        ValueOrStmt VisitUnaryPreInc(clang::UnaryOperator *expr);
        ValueOrStmt VisitUnaryPreDec(clang::UnaryOperator *expr);
        ValueOrStmt VisitUnaryAddrOf(clang::UnaryOperator *expr);
        ValueOrStmt VisitUnaryDeref(clang::UnaryOperator *expr);

        ValueOrStmt VisitUnaryPlus(clang::UnaryOperator *expr);
        ValueOrStmt VisitUnaryMinus(clang::UnaryOperator *expr);
        ValueOrStmt VisitUnaryNot(clang::UnaryOperator *expr);
        ValueOrStmt VisitUnaryLNot(clang::UnaryOperator *expr);
        ValueOrStmt VisitUnaryReal(clang::UnaryOperator *expr);
        ValueOrStmt VisitUnaryImag(clang::UnaryOperator *expr);
        ValueOrStmt VisitUnaryExtension(clang::UnaryOperator *expr);
        ValueOrStmt VisitUnaryCoawait(clang::UnaryOperator *expr);

        // Assembly Statements

        ValueOrStmt VisitAsmStmt(clang::AsmStmt *stmt);
        ValueOrStmt VisitGCCAsmStmt(clang::GCCAsmStmt *stmt);
        ValueOrStmt VisitMSAsmStmt(clang::MSAsmStmt *stmt);
        ValueOrStmt VisitCoroutineBodyStmt(clang::CoroutineBodyStmt *stmt);
        ValueOrStmt VisitCoreturnStmt(clang::CoreturnStmt *stmt);
        ValueOrStmt VisitCoroutineSuspendExpr(clang::CoroutineSuspendExpr *expr);
        ValueOrStmt VisitCoawaitExpr(clang::CoawaitExpr *expr);
        ValueOrStmt VisitCoyieldExpr(clang::CoyieldExpr *expr);
        ValueOrStmt VisitDependentCoawaitExpr(clang::DependentCoawaitExpr *expr);
        ValueOrStmt VisitAttributedStmt(clang::AttributedStmt *stmt);

        // Statements

        ValueOrStmt VisitBreakStmt(clang::BreakStmt *stmt);
        ValueOrStmt VisitCXXCatchStmt(clang::CXXCatchStmt *stmt);
        ValueOrStmt VisitCXXForRangeStmt(clang::CXXForRangeStmt *stmt);
        ValueOrStmt VisitCXXTryStmt(clang::CXXTryStmt *stmt);
        ValueOrStmt VisitCapturedStmt(clang::CapturedStmt *stmt);
        ValueOrStmt VisitCompoundStmt(clang::CompoundStmt *stmt);
        ValueOrStmt VisitContinueStmt(clang::ContinueStmt *stmt);
        ValueOrStmt VisitDeclStmt(clang::DeclStmt *stmt);
        ValueOrStmt VisitDoStmt(clang::DoStmt *stmt);

        // Expressions

        ValueOrStmt VisitAbstractConditionalOperator(clang::AbstractConditionalOperator *stmt);
        ValueOrStmt VisitBinaryConditionalOperator(clang::BinaryConditionalOperator *stmt);
        ValueOrStmt VisitConditionalOperator(clang::ConditionalOperator *stmt);
        ValueOrStmt VisitAddrLabelExpr(clang::AddrLabelExpr *expr);
        ValueOrStmt VisitConstantExpr(clang::ConstantExpr *expr);
        ValueOrStmt VisitArraySubscriptExpr(clang::ArraySubscriptExpr *expr);
        ValueOrStmt VisitArrayTypeTraitExpr(clang::ArrayTypeTraitExpr *expr);
        ValueOrStmt VisitAsTypeExpr(clang::AsTypeExpr *expr);
        ValueOrStmt VisitAtomicExpr(clang::AtomicExpr *expr);
        ValueOrStmt VisitBlockExpr(clang::BlockExpr *expr);
        ValueOrStmt VisitCXXBindTemporaryExpr(clang::CXXBindTemporaryExpr *expr);
        ValueOrStmt VisitCXXBoolLiteralExpr(const clang::CXXBoolLiteralExpr *lit);
        ValueOrStmt VisitCXXConstructExpr(clang::CXXConstructExpr *expr);
        ValueOrStmt VisitCXXTemporaryObjectExpr(clang::CXXTemporaryObjectExpr *expr);
        ValueOrStmt VisitCXXDefaultArgExpr(clang::CXXDefaultArgExpr *expr);
        ValueOrStmt VisitCXXDefaultInitExpr(clang::CXXDefaultInitExpr *expr);
        ValueOrStmt VisitCXXDeleteExpr(clang::CXXDeleteExpr *expr);
        ValueOrStmt VisitCXXDependentScopeMemberExpr(clang::CXXDependentScopeMemberExpr *expr);
        ValueOrStmt VisitCXXNewExpr(clang::CXXNewExpr *expr);
        ValueOrStmt VisitCXXNoexceptExpr(clang::CXXNoexceptExpr *expr);
        ValueOrStmt VisitCXXNullPtrLiteralExpr(clang::CXXNullPtrLiteralExpr *expr);
        ValueOrStmt VisitCXXPseudoDestructorExpr(clang::CXXPseudoDestructorExpr *expr);
        ValueOrStmt VisitCXXScalarValueInitExpr(clang::CXXScalarValueInitExpr *expr);
        ValueOrStmt VisitCXXStdInitializerListExpr(clang::CXXStdInitializerListExpr *expr);
        ValueOrStmt VisitCXXThisExpr(clang::CXXThisExpr *expr);
        ValueOrStmt VisitCXXThrowExpr(clang::CXXThrowExpr *expr);
        ValueOrStmt VisitCXXTypeidExpr(clang::CXXTypeidExpr *expr);
        ValueOrStmt VisitCXXFoldExpr(clang::CXXFoldExpr *expr);
        ValueOrStmt VisitCXXUnresolvedConstructExpr(clang::CXXUnresolvedConstructExpr *expr);
        ValueOrStmt VisitCXXUuidofExpr(clang::CXXUuidofExpr *expr);

        mlir::FuncOp VisitDirectCallee(clang::FunctionDecl *callee);
        mlir::Value VisitIndirectCallee(clang::Expr *callee);

        using Arguments = llvm::SmallVector< Value, 2 >;
        Arguments VisitArguments(clang::CallExpr *expr);
        ValueOrStmt VisitDirectCall(clang::CallExpr *expr);

        ValueOrStmt VisitIndirectCall(clang::CallExpr *expr);
        ValueOrStmt VisitCallExpr(clang::CallExpr *expr);

        ValueOrStmt VisitCUDAKernelCallExpr(clang::CUDAKernelCallExpr *expr);
        ValueOrStmt VisitCXXMemberCallExpr(clang::CXXMemberCallExpr *expr);
        ValueOrStmt VisitCXXOperatorCallExpr(clang::CXXOperatorCallExpr *expr);

        ValueOrStmt VisitUserDefinedLiteral(clang::UserDefinedLiteral *lit);
        ValueOrStmt VisitCStyleCastExpr(clang::CStyleCastExpr *expr);
        ValueOrStmt VisitCXXFunctionalCastExpr(clang::CXXFunctionalCastExpr *expr);
        ValueOrStmt VisitCXXConstCastExpr(clang::CXXConstCastExpr *expr);
        ValueOrStmt VisitCXXDynamicCastExpr(clang::CXXDynamicCastExpr *expr);
        ValueOrStmt VisitCXXReinterpretCastExpr(clang::CXXReinterpretCastExpr *expr);
        ValueOrStmt VisitCXXStaticCastExpr(clang::CXXStaticCastExpr *expr);

        ValueOrStmt VisitObjCBridgedCastExpr(clang::ObjCBridgedCastExpr *expr);
        ValueOrStmt VisitImplicitCastExpr(clang::ImplicitCastExpr *expr);
        ValueOrStmt VisitCharacterLiteral(clang::CharacterLiteral *lit);
        ValueOrStmt VisitChooseExpr(clang::ChooseExpr *expr);
        ValueOrStmt VisitCompoundLiteralExpr(clang::CompoundLiteralExpr *expr);
        ValueOrStmt VisitConvertVectorExpr(clang::ConvertVectorExpr *expr);
        ValueOrStmt VisitDeclRefExpr(clang::DeclRefExpr *expr);
        ValueOrStmt VisitDependentScopeDeclRefExpr(clang::DependentScopeDeclRefExpr *expr);

        ValueOrStmt VisitDesignatedInitExpr(clang::DesignatedInitExpr *expr);
        ValueOrStmt VisitExprWithCleanups(clang::ExprWithCleanups *expr);
        ValueOrStmt VisitExpressionTraitExpr(clang::ExpressionTraitExpr *expr);
        ValueOrStmt VisitExtVectorElementExpr(clang::ExtVectorElementExpr *expr);
        ValueOrStmt VisitFloatingLiteral(clang::FloatingLiteral *lit);
        ValueOrStmt VisitFunctionParmPackExpr(clang::FunctionParmPackExpr *expr);
        ValueOrStmt VisitGNUNullExpr(clang::GNUNullExpr *expr);
        ValueOrStmt VisitGenericSelectionExpr(clang::GenericSelectionExpr *expr);
        ValueOrStmt VisitImaginaryLiteral(clang::ImaginaryLiteral *lit);
        ValueOrStmt VisitFixedPointLiteral(clang::FixedPointLiteral *lit);
        ValueOrStmt VisitImplicitValueInitExpr(clang::ImplicitValueInitExpr *expr);
        ValueOrStmt VisitInitListExpr(clang::InitListExpr *expr);
        ValueOrStmt VisitIntegerLiteral(const clang::IntegerLiteral *lit);
        ValueOrStmt VisitLambdaExpr(clang::LambdaExpr *expr);
        ValueOrStmt VisitMSPropertyRefExpr(clang::MSPropertyRefExpr *expr);
        ValueOrStmt VisitMaterializeTemporaryExpr(clang::MaterializeTemporaryExpr *expr);
        ValueOrStmt VisitMemberExpr(clang::MemberExpr *expr);

        ValueOrStmt VisitObjCArrayLiteral(clang::ObjCArrayLiteral *expr);
        ValueOrStmt VisitObjCBoolLiteralExpr(clang::ObjCBoolLiteralExpr *expr);
        ValueOrStmt VisitObjCBoxedExpr(clang::ObjCBoxedExpr *expr);
        ValueOrStmt VisitObjCDictionaryLiteral(clang::ObjCDictionaryLiteral *lit);
        ValueOrStmt VisitObjCEncodeExpr(clang::ObjCEncodeExpr *expr);
        ValueOrStmt VisitObjCIndirectCopyRestoreExpr(clang::ObjCIndirectCopyRestoreExpr *expr);
        ValueOrStmt VisitObjCIsaExpr(clang::ObjCIsaExpr *expr);
        ValueOrStmt VisitObjCIvarRefExpr(clang::ObjCIvarRefExpr *expr);
        ValueOrStmt VisitObjCMessageExpr(clang::ObjCMessageExpr *expr);
        ValueOrStmt VisitObjCPropertyRefExpr(clang::ObjCPropertyRefExpr *expr);
        ValueOrStmt VisitObjCProtocolExpr(clang::ObjCProtocolExpr *expr);
        ValueOrStmt VisitObjCSelectorExpr(clang::ObjCSelectorExpr *expr);
        ValueOrStmt VisitObjCStringLiteral(clang::ObjCStringLiteral *lit);

        ValueOrStmt VisitObjCSubscriptRefExpr(clang::ObjCSubscriptRefExpr *expr);

        ValueOrStmt VisitOffsetOfExpr(clang::OffsetOfExpr *expr);

        ValueOrStmt VisitOpaqueValueExpr(clang::OpaqueValueExpr *expr);
        ValueOrStmt VisitOverloadExpr(clang::OverloadExpr *expr);
        ValueOrStmt VisitUnresolvedLookupExpr(clang::UnresolvedLookupExpr *expr);
        ValueOrStmt VisitUnresolvedMemberExpr(clang::UnresolvedMemberExpr *expr);
        ValueOrStmt VisitPackExpansionExpr(clang::PackExpansionExpr *expr);
        ValueOrStmt VisitParenExpr(clang::ParenExpr *expr);

        ValueOrStmt VisitParenListExpr(clang::ParenListExpr *expr);
        ValueOrStmt VisitPredefinedExpr(clang::PredefinedExpr *expr);
        ValueOrStmt VisitPseudoObjectExpr(clang::PseudoObjectExpr *expr);
        ValueOrStmt VisitShuffleVectorExpr(clang::ShuffleVectorExpr *expr);
        ValueOrStmt VisitSizeOfPackExpr(clang::SizeOfPackExpr *expr);

        ValueOrStmt VisitStmtExpr(clang::StmtExpr *expr);

        ValueOrStmt VisitStringLiteral(clang::StringLiteral *lit);

        ValueOrStmt VisitSubstNonTypeTemplateParmExpr(clang::SubstNonTypeTemplateParmExpr *expr);

        ValueOrStmt VisitSubstNonTypeTemplateParmPackExpr(
            clang::SubstNonTypeTemplateParmPackExpr *expr);

        ValueOrStmt VisitTypeTraitExpr(clang::TypeTraitExpr *expr);
        ValueOrStmt VisitUnaryExprOrTypeTraitExpr(clang::UnaryExprOrTypeTraitExpr *expr);

        ValueOrStmt VisitSourceLocExpr(clang::SourceLocExpr *expr);
        ValueOrStmt VisitVAArgExpr(clang::VAArgExpr *expr);

        // Statements

        ValueOrStmt VisitForStmt(clang::ForStmt *stmt);
        ValueOrStmt VisitGotoStmt(clang::GotoStmt *stmt);
        ValueOrStmt VisitIfStmt(clang::IfStmt *stmt);
        ValueOrStmt VisitIndirectGotoStmt(clang::IndirectGotoStmt *stmt);
        ValueOrStmt VisitLabelStmt(clang::LabelStmt *stmt);
        ValueOrStmt VisitMSDependentExistsStmt(clang::MSDependentExistsStmt *stmt);
        ValueOrStmt VisitNullStmt(clang::NullStmt *stmt);
        ValueOrStmt VisitOMPBarrierDirective(clang::OMPBarrierDirective *dir);
        ValueOrStmt VisitOMPCriticalDirective(clang::OMPCriticalDirective *dir);
        ValueOrStmt VisitOMPFlushDirective(clang::OMPFlushDirective *dir);
        ValueOrStmt VisitOMPForDirective(clang::OMPForDirective *dir);
        ValueOrStmt VisitOMPMasterDirective(clang::OMPMasterDirective *dir);
        ValueOrStmt VisitOMPParallelDirective(clang::OMPParallelDirective *dir);
        ValueOrStmt VisitOMPParallelForDirective(clang::OMPParallelForDirective *dir);
        ValueOrStmt VisitOMPParallelSectionsDirective(clang::OMPParallelSectionsDirective *dir);
        ValueOrStmt VisitOMPSectionDirective(clang::OMPSectionDirective *dir);
        ValueOrStmt VisitOMPSectionsDirective(clang::OMPSectionsDirective *dir);
        ValueOrStmt VisitOMPSimdDirective(clang::OMPSimdDirective *dir);
        ValueOrStmt VisitOMPSingleDirective(clang::OMPSingleDirective *dir);
        ValueOrStmt VisitOMPTaskDirective(clang::OMPTaskDirective *dir);
        ValueOrStmt VisitOMPTaskwaitDirective(clang::OMPTaskwaitDirective *dir);
        ValueOrStmt VisitOMPTaskyieldDirective(clang::OMPTaskyieldDirective *dir);
        ValueOrStmt VisitObjCAtCatchStmt(clang::ObjCAtCatchStmt *stmt);
        ValueOrStmt VisitObjCAtFinallyStmt(clang::ObjCAtFinallyStmt *stmt);
        ValueOrStmt VisitObjCAtSynchronizedStmt(clang::ObjCAtSynchronizedStmt *stmt);
        ValueOrStmt VisitObjCAtThrowStmt(clang::ObjCAtThrowStmt *stmt);
        ValueOrStmt VisitObjCAtTryStmt(clang::ObjCAtTryStmt *stmt);
        ValueOrStmt VisitObjCAutoreleasePoolStmt(clang::ObjCAutoreleasePoolStmt *stmt);
        ValueOrStmt VisitObjCForCollectionStmt(clang::ObjCForCollectionStmt *stmt);
        ValueOrStmt VisitReturnStmt(clang::ReturnStmt *stmt);
        ValueOrStmt VisitSEHExceptStmt(clang::SEHExceptStmt *stmt);
        ValueOrStmt VisitSEHFinallyStmt(clang::SEHFinallyStmt *stmt);
        ValueOrStmt VisitSEHLeaveStmt(clang::SEHLeaveStmt *stmt);
        ValueOrStmt VisitSEHTryStmt(clang::SEHTryStmt *stmt);
        ValueOrStmt VisitCaseStmt(clang::CaseStmt *stmt);
        ValueOrStmt VisitDefaultStmt(clang::DefaultStmt *stmt);
        ValueOrStmt VisitSwitchStmt(clang::SwitchStmt *stmt);
        ValueOrStmt VisitWhileStmt(clang::WhileStmt *stmt);
        ValueOrStmt VisitBuiltinBitCastExpr(clang::BuiltinBitCastExpr *expr);

        // Declarations

        ValueOrStmt VisitImportDecl(clang::ImportDecl *decl);
        ValueOrStmt VisitEmptyDecl(clang::EmptyDecl *decl);
        ValueOrStmt VisitAccessSpecDecl(clang::AccessSpecDecl *decl);
        ValueOrStmt VisitCapturedDecl(clang::CapturedDecl *decl);
        ValueOrStmt VisitClassScopeFunctionSpecializationDecl(
            clang::ClassScopeFunctionSpecializationDecl *decl);
        ValueOrStmt VisitExportDecl(clang::ExportDecl *decl);

        ValueOrStmt VisitExternCContextDecl(clang::ExternCContextDecl *decl);
        ValueOrStmt VisitFileScopeAsmDecl(clang::FileScopeAsmDecl *decl);
        ValueOrStmt VisitStaticAssertDecl(clang::StaticAssertDecl *decl);
        ValueOrStmt VisitTranslationUnitDecl(clang::TranslationUnitDecl *decl);
        ValueOrStmt VisitBindingDecl(clang::BindingDecl *decl);
        // ValueOrStmt VisitNamespaceDecl(clang::NamespaceDecl *decl);

        ValueOrStmt VisitNamespaceAliasDecl(clang::NamespaceAliasDecl *decl);
        // ValueOrStmt VisitTypedefNameDecl(clang::TypedefNameDecl *decl);

        ValueOrStmt VisitTypedefDecl(clang::TypedefDecl *decl);
        ValueOrStmt VisitTypeAliasDecl(clang::TypeAliasDecl *decl);
        ValueOrStmt VisitTemplateDecl(clang::TemplateDecl *decl);
        ValueOrStmt VisitTypeAliasTemplateDecl(clang::TypeAliasTemplateDecl *decl);
        ValueOrStmt VisitLabelDecl(clang::LabelDecl *decl);
        ValueOrStmt VisitEnumDecl(clang::EnumDecl *decl);
        ValueOrStmt VisitRecordDecl(clang::RecordDecl *decl);
        ValueOrStmt VisitEnumConstantDecl(clang::EnumConstantDecl *decl);
        ValueOrStmt VisitFunctionDecl(clang::FunctionDecl *decl);
        ValueOrStmt VisitCXXMethodDecl(clang::CXXMethodDecl *decl);
        ValueOrStmt VisitCXXConstructorDecl(clang::CXXConstructorDecl *decl);
        ValueOrStmt VisitCXXDestructorDecl(clang::CXXDestructorDecl *decl);
        ValueOrStmt VisitCXXConversionDecl(clang::CXXConversionDecl *decl);
        ValueOrStmt VisitCXXDeductionGuideDecl(clang::CXXDeductionGuideDecl *decl);
        ValueOrStmt VisitMSPropertyDecl(clang::MSPropertyDecl *decl);
        ValueOrStmt VisitMSGuidDecl(clang::MSGuidDecl *decl);
        ValueOrStmt VisitFieldDecl(clang::FieldDecl *decl);
        ValueOrStmt VisitIndirectFieldDecl(clang::IndirectFieldDecl *decl);
        ValueOrStmt VisitFriendDecl(clang::FriendDecl *decl);
        ValueOrStmt VisitFriendTemplateDecl(clang::FriendTemplateDecl *decl);
        ValueOrStmt VisitObjCAtDefsFieldDecl(clang::ObjCAtDefsFieldDecl *decl);
        ValueOrStmt VisitObjCIvarDecl(clang::ObjCIvarDecl *decl);
        ValueOrStmt VisitVarDecl(clang::VarDecl *decl);
        ValueOrStmt VisitDecompositionDecl(clang::DecompositionDecl *decl);
        ValueOrStmt VisitImplicitParamDecl(clang::ImplicitParamDecl *decl);
        // ValueOrStmt VisitUnresolvedUsingIfExistsDecl(clang::UnresolvedUsingIfExistsDecl *decl);
        ValueOrStmt VisitParmVarDecl(clang::ParmVarDecl *decl);
        ValueOrStmt VisitObjCMethodDecl(clang::ObjCMethodDecl *decl);
        ValueOrStmt VisitObjCTypeParamDecl(clang::ObjCTypeParamDecl *decl);
        ValueOrStmt VisitObjCProtocolDecl(clang::ObjCProtocolDecl *decl);
        ValueOrStmt VisitLinkageSpecDecl(clang::LinkageSpecDecl *decl);
        ValueOrStmt VisitUsingDecl(clang::UsingDecl *decl);
        ValueOrStmt VisitUsingShadowDecl(clang::UsingShadowDecl *decl);
        ValueOrStmt VisitUsingDirectiveDecl(clang::UsingDirectiveDecl *decl);
        ValueOrStmt VisitUsingPackDecl(clang::UsingPackDecl *decl);
        // ValueOrStmt VisitUsingEnumDecl(clang::UsingEnumDecl *decl);
        ValueOrStmt VisitUnresolvedUsingValueDecl(clang::UnresolvedUsingValueDecl *decl);
        ValueOrStmt VisitUnresolvedUsingTypenameDecl(clang::UnresolvedUsingTypenameDecl *decl);
        ValueOrStmt VisitBuiltinTemplateDecl(clang::BuiltinTemplateDecl *decl);
        ValueOrStmt VisitConceptDecl(clang::ConceptDecl *decl);
        ValueOrStmt VisitRedeclarableTemplateDecl(clang::RedeclarableTemplateDecl *decl);
        ValueOrStmt VisitLifetimeExtendedTemporaryDecl(clang::LifetimeExtendedTemporaryDecl *decl);
        ValueOrStmt VisitPragmaCommentDecl(clang::PragmaCommentDecl *decl);
        ValueOrStmt VisitPragmaDetectMismatchDecl(clang::PragmaDetectMismatchDecl *decl);
        ValueOrStmt VisitRequiresExprBodyDecl(clang::RequiresExprBodyDecl *decl);
        ValueOrStmt VisitObjCCompatibleAliasDecl(clang::ObjCCompatibleAliasDecl *decl);
        ValueOrStmt VisitObjCCategoryDecl(clang::ObjCCategoryDecl *decl);
        ValueOrStmt VisitObjCImplDecl(clang::ObjCImplDecl *decl);
        ValueOrStmt VisitObjCInterfaceDecl(clang::ObjCInterfaceDecl *decl);
        ValueOrStmt VisitObjCCategoryImplDecl(clang::ObjCCategoryImplDecl *decl);
        ValueOrStmt VisitObjCImplementationDecl(clang::ObjCImplementationDecl *decl);
        ValueOrStmt VisitObjCPropertyDecl(clang::ObjCPropertyDecl *decl);
        ValueOrStmt VisitObjCPropertyImplDecl(clang::ObjCPropertyImplDecl *decl);
        ValueOrStmt VisitTemplateParamObjectDecl(clang::TemplateParamObjectDecl *decl);
        ValueOrStmt VisitTemplateTypeParmDecl(clang::TemplateTypeParmDecl *decl);
        ValueOrStmt VisitNonTypeTemplateParmDecl(clang::NonTypeTemplateParmDecl *decl);
        ValueOrStmt VisitTemplateTemplateParmDecl(clang::TemplateTemplateParmDecl *decl);
        ValueOrStmt VisitClassTemplateDecl(clang::ClassTemplateDecl *decl);
        ValueOrStmt VisitClassTemplatePartialSpecializationDecl(
            clang::ClassTemplatePartialSpecializationDecl *decl);
        ValueOrStmt VisitClassTemplateSpecializationDecl(
            clang::ClassTemplateSpecializationDecl *decl);
        ValueOrStmt VisitVarTemplateDecl(clang::VarTemplateDecl *decl);
        ValueOrStmt VisitVarTemplateSpecializationDecl(clang::VarTemplateSpecializationDecl *decl);
        ValueOrStmt VisitVarTemplatePartialSpecializationDecl(
            clang::VarTemplatePartialSpecializationDecl *decl);
        ValueOrStmt VisitFunctionTemplateDecl(clang::FunctionTemplateDecl *decl);
        ValueOrStmt VisitConstructorUsingShadowDecl(clang::ConstructorUsingShadowDecl *decl);

        ValueOrStmt VisitOMPAllocateDecl(clang::OMPAllocateDecl *decl);
        ValueOrStmt VisitOMPRequiresDecl(clang::OMPRequiresDecl *decl);
        ValueOrStmt VisitOMPThreadPrivateDecl(clang::OMPThreadPrivateDecl *decl);
        ValueOrStmt VisitOMPCapturedExprDecl(clang::OMPCapturedExprDecl *decl);
        ValueOrStmt VisitOMPDeclareReductionDecl(clang::OMPDeclareReductionDecl *decl);
        ValueOrStmt VisitOMPDeclareMapperDecl(clang::OMPDeclareMapperDecl *decl);

      private:
        template< typename Op >
        ValueOrStmt make_bin(clang::BinaryOperator *expr) {
            auto lhs = Visit(expr->getLHS());
            auto rhs = Visit(expr->getRHS());
            auto loc = builder.get_end_location(expr->getSourceRange());
            auto res = builder.make< Op >(loc, lhs, rhs);

            if constexpr (std::is_convertible_v< decltype(res), Value >) {
                return Value(res);
            } else {
                return res;
            }
        }

        template< typename Op >
        ValueOrStmt make_ibin(clang::BinaryOperator *expr) {
            auto ty = expr->getType();
            if (ty->isIntegerType())
                return make_bin< Op >(expr);
            return Value();
        }

        template< typename UOp, typename SOp >
        ValueOrStmt make_ibin(clang::BinaryOperator *expr) {
            auto ty = expr->getType();
            if (ty->isUnsignedIntegerType())
                return make_bin< UOp >(expr);
            if (ty->isIntegerType())
                return make_bin< SOp >(expr);
            return Value();
        }

        template< Predicate pred >
        Value make_cmp(clang::BinaryOperator *expr) {
            auto lhs = Visit(expr->getLHS());
            auto rhs = Visit(expr->getRHS());
            auto loc = builder.get_end_location(expr->getSourceRange());
            auto res = types.convert(expr->getType());
            return builder.make_value< CmpOp >(loc, res, pred, lhs, rhs);
        }

        template< Predicate pred >
        Value make_icmp(clang::BinaryOperator *expr) {
            auto ty = expr->getLHS()->getType();
            if (ty->isIntegerType())
                return make_cmp< pred >(expr);
            return Value();
        }

        template< Predicate upred, Predicate spred >
        Value make_icmp(clang::BinaryOperator *expr) {
            auto ty = expr->getLHS()->getType();
            if (ty->isUnsignedIntegerType())
                return make_cmp< upred >(expr);
            if (ty->isIntegerType())
                return make_cmp< spred >(expr);
            return Value();
        }

        template< typename Op >
        ValueOrStmt make_type_preserving_unary(clang::UnaryOperator *expr) {
            auto loc = builder.get_end_location(expr->getSourceRange());
            auto arg = Visit(expr->getSubExpr());
            return builder.make_value< Op >(loc, arg);
        }

        template< typename Op >
        ValueOrStmt make_unary(clang::UnaryOperator *expr) {
            auto loc = builder.get_location(expr->getSourceRange());
            auto rty = types.convert(expr->getType());
            auto arg = Visit(expr->getSubExpr());
            return builder.make_value< Op >(loc, rty, arg);
        }

        template< typename Cast >
        ValueOrStmt make_cast(clang::CastExpr *expr) {
            auto loc = builder.get_location(expr->getSourceRange());
            auto rty = types.convert(expr->getType());
            auto arg = Visit(expr->getSubExpr());
            return builder.make_value< Cast >(loc, rty, arg, cast_kind(expr));
        }

        inline auto make_region_builder(clang::Stmt *stmt) {
            return [stmt, this](auto &bld, auto) {
                if (stmt) {
                    Visit(stmt);
                }
                splice_trailing_scopes(*bld.getBlock()->getParent());
            };
        }

        inline auto make_cond_builder(clang::Stmt *stmt) {
            return [stmt, this](auto &bld, auto loc) {
                Visit(stmt);
                auto &op = bld.getBlock()->back();
                assert(op.getNumResults() == 1);
                auto cond = op.getResult(0);
                bld.template create< CondYieldOp >(loc, cond);
            };
        }

        inline auto make_value_builder(clang::Stmt *stmt) {
            return [stmt, this](auto &bld, auto loc) {
                Visit(stmt);
                auto &op = bld.getBlock()->back();
                assert(op.getNumResults() == 1);
                auto cond = op.getResult(0);
                bld.template create< ValueYieldOp >(loc, cond);
            };
        }

        inline auto make_yield_true() {
            return [this](auto &bld, auto loc) {
                auto t = builder.true_value(loc);
                bld.template create< CondYieldOp >(loc, t);
            };
        }

        template< typename LiteralType >
        inline auto make_scalar_literal(LiteralType *lit) {
            auto type = types.convert(lit->getType());
            auto loc  = builder.get_location(lit->getSourceRange());
            return builder.constant(loc, type, lit->getValue());
        }

        inline void walk_type(clang::QualType type, invocable< clang::Type * > auto &&yield) {
            if (yield(type)) {
                return;
            }

            if (auto arr = clang::dyn_cast< clang::ArrayType >(type)) {
                walk_type(arr->getElementType(), yield);
            }

            if (auto ptr = clang::dyn_cast< clang::PointerType >(type)) {
                walk_type(ptr->getPointeeType(), yield);
            }
        }

        inline ValueOrStmt checked(ValueOrStmt &&val) {
            VAST_CHECK(check(val), "unsupported operation type");
            return val;
        }

        template< typename Op >
        auto make_expr_trait_expr(clang::UnaryExprOrTypeTraitExpr *expr, auto rty, auto loc) {
            auto arg = make_value_builder(expr->getArgumentExpr());
            return builder.make_value< Op >(loc, rty, arg);
        }

        template< typename Op >
        auto make_type_trait_expr(clang::UnaryExprOrTypeTraitExpr *expr, auto rty, auto loc) {
            auto arg = types.convert(expr->getArgumentType());
            return builder.make_value< Op >(loc, rty, arg);
        }

        template< typename TypeTraitOp, typename ExprTraitOp >
        auto dispatch_trait_expr(clang::UnaryExprOrTypeTraitExpr *expr) {
            auto loc = builder.get_location(expr->getSourceRange());
            auto rty = types.convert(expr->getType());

            return expr->isArgumentType() ? make_type_trait_expr< TypeTraitOp >(expr, rty, loc)
                                          : make_expr_trait_expr< ExprTraitOp >(expr, rty, loc);
        }

        template< typename Var, typename... Args >
        Var make_var(clang::VarDecl *decl, Args &&...args) {
            return builder.make< Var >(std::forward< Args >(args)...);
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

        TranslationContext &ctx;
        HighLevelBuilder builder;
        HighLevelTypeConverter types;
    };
} // namespace vast::hl
