// Copyright (c) 2024-present, Trail of Bits, Inc.

#ifndef VAST_INTERFACES_AST_EXPR_INTERFACE
#define VAST_INTERFACES_AST_EXPR_INTERFACE

include "mlir/IR/OpBase.td"
include "vast/Interfaces/AST/Common.td"
include "vast/Interfaces/AST/StmtInterface.td"

def VAST_ExprInterface : VAST_ASTOpInterface< "ExprInterface",
    [VAST_ValueStmtInterface] > {

    let description = [{
        An interface to provide [`clang::Expr`]
        (https://clang.llvm.org/doxygen/classclang_1_1Expr.html) functionality.
    }];

    let methods = [
        InterfaceMethod< "clang::Expr::getType", "::vast::ast::QualTypeInterface", "getType" >,
        InterfaceMethod< "clang::Expr::IgnoreParenNoopCasts", "::vast::ast::ExprInterface",
                         "IgnoreParenNoopCasts", (ins "::vast::ast::ASTContextInterface":$Ctx) >,
        InterfaceMethod< "clang::Expr::IgnoreParens", "::vast::ast::ExprInterface", "IgnoreParens" >
    ];
}

def VAST_PredefinedExprInterface : VAST_ASTOpInterface< "PredefinedExprInterface",
    [VAST_ExprInterface] > {

    let description = [{
        An interface to provide [`clang::PredefinedExpr`]
        (https://clang.llvm.org/doxygen/classclang_1_1PredefinedExpr.html) functionality.
    }];

    let methods = [];
}

def VAST_SYCLUniqueStableNameExprInterface : VAST_ASTOpInterface< "SYCLUniqueStableNameExprInterface",
    [VAST_ExprInterface] > {

    let description = [{
        An interface to provide [`clang::SYCLUniqueStableNameExpr`]
        (https://clang.llvm.org/doxygen/classclang_1_1SYCLUniqueStableNameExpr.html) functionality.
    }];

    let methods = [];
}

def VAST_DeclRefExprInterface : VAST_ASTOpInterface< "DeclRefExprInterface",
    [VAST_ExprInterface] > {

    let description = [{
        An interface to provide [`clang::DeclRefExpr`]
        (https://clang.llvm.org/doxygen/classclang_1_1DeclRefExpr.html) functionality.
    }];

    let methods = [];
}

def VAST_IntegerLiteralInterface : VAST_ASTOpInterface< "IntegerLiteralInterface",
    [VAST_ExprInterface] > {

    let description = [{
        An interface to provide [`clang::IntegerLiteral`]
        (https://clang.llvm.org/doxygen/classclang_1_1IntegerLiteral.html) functionality.
    }];

    let methods = [];
}

def VAST_FixedPointLiteralInterface : VAST_ASTOpInterface< "FixedPointLiteralInterface",
    [VAST_ExprInterface] > {

    let description = [{
        An interface to provide [`clang::FixedPointLiteral`]
        (https://clang.llvm.org/doxygen/classclang_1_1FixedPointLiteral.html) functionality.
    }];

    let methods = [];
}

def VAST_FloatingLiteralInterface : VAST_ASTOpInterface< "FloatingLiteralInterface",
    [VAST_ExprInterface] > {

    let description = [{
        An interface to provide [`clang::FloatingLiteral`]
        (https://clang.llvm.org/doxygen/classclang_1_1FloatingLiteral.html) functionality.
    }];

    let methods = [];
}

def VAST_ImaginaryLiteralInterface : VAST_ASTOpInterface< "ImaginaryLiteralInterface",
    [VAST_ExprInterface] > {

    let description = [{
        An interface to provide [`clang::ImaginaryLiteral`]
        (https://clang.llvm.org/doxygen/classclang_1_1ImaginaryLiteral.html) functionality.
    }];

    let methods = [];
}

def VAST_StringLiteralInterface : VAST_ASTOpInterface< "StringLiteralInterface",
    [VAST_ExprInterface] > {

    let description = [{
        An interface to provide [`clang::StringLiteral`]
        (https://clang.llvm.org/doxygen/classclang_1_1StringLiteral.html) functionality.
    }];

    let methods = [];
}

def VAST_CharacterLiteralInterface : VAST_ASTOpInterface< "CharacterLiteralInterface",
    [VAST_ExprInterface] > {

    let description = [{
        An interface to provide [`clang::CharacterLiteral`]
        (https://clang.llvm.org/doxygen/classclang_1_1CharacterLiteral.html) functionality.
    }];

    let methods = [];
}

def VAST_ParenExprInterface : VAST_ASTOpInterface< "ParenExprInterface",
    [VAST_ExprInterface] > {

    let description = [{
        An interface to provide [`clang::ParenExpr`]
        (https://clang.llvm.org/doxygen/classclang_1_1ParenExpr.html) functionality.
    }];

    let methods = [];
}

def VAST_UnaryOperatorInterface : VAST_ASTOpInterface< "UnaryOperatorInterface",
    [VAST_ExprInterface] > {

    let description = [{
        An interface to provide [`clang::UnaryOperator`]
        (https://clang.llvm.org/doxygen/classclang_1_1UnaryOperator.html) functionality.
    }];

    let methods = [
        InterfaceMethod< "clang::UnaryOperator::getOpcode", "::clang::UnaryOperatorKind", "getOpcode" >,
        InterfaceMethod< "clang::UnaryOperator::getSubExpr", "::vast::ast::ExprInterface", "getSubExpr" >,
        InterfaceMethod< "clang::UnaryOperator::isIncrementDecrementOp", "bool", "isIncrementDecrementOp" >
    ];
}

def VAST_OffsetOfExprInterface : VAST_ASTOpInterface< "OffsetOfExprInterface",
    [VAST_ExprInterface] > {

    let description = [{
        An interface to provide [`clang::OffsetOfExpr`]
        (https://clang.llvm.org/doxygen/classclang_1_1OffsetOfExpr.html) functionality.
    }];

    let methods = [];
}

def VAST_UnaryExprOrTypeTraitExprInterface : VAST_ASTOpInterface< "UnaryExprOrTypeTraitExprInterface",
    [VAST_ExprInterface] > {

    let description = [{
        An interface to provide [`clang::UnaryExprOrTypeTraitExpr`]
        (https://clang.llvm.org/doxygen/classclang_1_1UnaryExprOrTypeTraitExpr.html) functionality.
    }];

    let methods = [];
}

def VAST_ArraySubscriptExprInterface : VAST_ASTOpInterface< "ArraySubscriptExprInterface",
    [VAST_ExprInterface] > {

    let description = [{
        An interface to provide [`clang::ArraySubscriptExpr`]
        (https://clang.llvm.org/doxygen/classclang_1_1ArraySubscriptExpr.html) functionality.
    }];

    let methods = [];
}

def VAST_MatrixSubscriptExprInterface : VAST_ASTOpInterface< "MatrixSubscriptExprInterface",
    [VAST_ExprInterface] > {

    let description = [{
        An interface to provide [`clang::MatrixSubscriptExpr`]
        (https://clang.llvm.org/doxygen/classclang_1_1MatrixSubscriptExpr.html) functionality.
    }];

    let methods = [];
}

def VAST_ArraySectionExprInterface : VAST_ASTOpInterface< "ArraySectionExprInterface",
    [VAST_ExprInterface] > {

    let description = [{
        An interface to provide [`clang::ArraySectionExpr`]
        (https://clang.llvm.org/doxygen/classclang_1_1ArraySectionExpr.html) functionality.
    }];

    let methods = [];
}

def VAST_OMPIteratorExprInterface : VAST_ASTOpInterface< "OMPIteratorExprInterface",
    [VAST_ExprInterface] > {

    let description = [{
        An interface to provide [`clang::OMPIteratorExpr`]
        (https://clang.llvm.org/doxygen/classclang_1_1OMPIteratorExpr.html) functionality.
    }];

    let methods = [];
}

def VAST_CallExprInterface : VAST_ASTOpInterface< "CallExprInterface",
    [VAST_ExprInterface] > {

    let description = [{
        An interface to provide [`clang::CallExpr`]
        (https://clang.llvm.org/doxygen/classclang_1_1CallExpr.html) functionality.
    }];

    let methods = [
        InterfaceMethod< "clang::CallExpr::isCallToStdMove", "bool", "isCallToStdMove" >,
        InterfaceMethod< "clang::CallExpr::getArg", "::vast::ast::ExprInterface", "getArg", (ins "unsigned":$arg) > 
    ];
}

def VAST_MemberExprInterface : VAST_ASTOpInterface< "MemberExprInterface",
    [VAST_ExprInterface] > {

    let description = [{
        An interface to provide [`clang::MemberExpr`]
        (https://clang.llvm.org/doxygen/classclang_1_1MemberExpr.html) functionality.
    }];

    let methods = [
        InterfaceMethod< "clang::MemberExpr::getBase", "::vast::ast::ExprInterface", "getBase" >
    ];
}

def VAST_CastExprInterface : VAST_ASTOpInterface< "CastExprInterface",
    [VAST_ExprInterface] > {

    let description = [{
        An interface to provide [`clang::CastExpr`]
        (https://clang.llvm.org/doxygen/classclang_1_1CastExpr.html) functionality.
    }];

    let methods = [
        InterfaceMethod< "clang::CastExpr::getCastKind", "::clang::CastKind", "getCastKind" >,
        InterfaceMethod< "clang::CastExpr::getSubExpr", "::vast::ast::ExprInterface", "getSubExpr" >
    ];
}

def VAST_BinaryOperatorInterface : VAST_ASTOpInterface< "BinaryOperatorInterface",
    [VAST_ExprInterface] > {

    let description = [{
        An interface to provide [`clang::BinaryOperator`]
        (https://clang.llvm.org/doxygen/classclang_1_1BinaryOperator.html) functionality.
    }];

    let methods = [
        InterfaceMethod< "clang::BinaryOperator::getOpcode", "::clang::BinaryOperatorKind", "getOpcode" >,
        InterfaceMethod< "clang::BinaryOperator::getLHS", "::vast::ast::ExprInterface", "getLHS" >,
        InterfaceMethod< "clang::BinaryOperator::getRHS", "::vast::ast::ExprInterface", "getRHS" >,
        InterfaceMethod< "clang::BinaryOperator::isCompoundAssignmentOp", "bool", "isCompoundAssignmentOp" >
    ];
}

def VAST_CompoundAssignOperatorInterface : VAST_ASTOpInterface< "CompoundAssignOperatorInterface",
    [VAST_BinaryOperatorInterface] > {

    let description = [{
        An interface to provide [`clang::CompoundAssignOperator`]
        (https://clang.llvm.org/doxygen/classclang_1_1CompoundAssignOperator.html) functionality.
    }];

    let methods = [];
}

def VAST_AbstractConditionalOperatorInterface : VAST_ASTOpInterface< "AbstractConditionalOperatorInterface",
    [VAST_ExprInterface] > {

    let description = [{
        An interface to provide [`clang::AbstractConditionalOperator`]
        (https://clang.llvm.org/doxygen/classclang_1_1AbstractConditionalOperator.html) functionality.
    }];

    let methods = [];
}

def VAST_ConditionalOperatorInterface : VAST_ASTOpInterface< "ConditionalOperatorInterface",
    [VAST_AbstractConditionalOperatorInterface] > {

    let description = [{
        An interface to provide [`clang::ConditionalOperator`]
        (https://clang.llvm.org/doxygen/classclang_1_1ConditionalOperator.html) functionality.
    }];

    let methods = [
        InterfaceMethod< "clang::ConditionalOperator::getTrueExpr", "::vast::ast::ExprInterface", "getTrueExpr" >,
        InterfaceMethod< "clang::ConditionalOperator::getFalseExpr", "::vast::ast::ExprInterface", "getFalseExpr" >
    ];
}

def VAST_BinaryConditionalOperatorInterface : VAST_ASTOpInterface< "BinaryConditionalOperatorInterface",
    [VAST_AbstractConditionalOperatorInterface] > {

    let description = [{
        An interface to provide [`clang::BinaryConditionalOperator`]
        (https://clang.llvm.org/doxygen/classclang_1_1BinaryConditionalOperator.html) functionality.
    }];

    let methods = [
        InterfaceMethod< "clang::BinaryConditionalOperator::getTrueExpr", "::vast::ast::ExprInterface", "getTrueExpr" >,
        InterfaceMethod< "clang::BinaryConditionalOperator::getFalseExpr", "::vast::ast::ExprInterface", "getFalseExpr" >
    ];
}

def VAST_ImplicitCastExprInterface : VAST_ASTOpInterface< "ImplicitCastExprInterface",
    [VAST_CastExprInterface] > {

    let description = [{
        An interface to provide [`clang::ImplicitCastExpr`]
        (https://clang.llvm.org/doxygen/classclang_1_1ImplicitCastExpr.html) functionality.
    }];

    let methods = [];
}

def VAST_ExplicitCastExprInterface : VAST_ASTOpInterface< "ExplicitCastExprInterface",
    [VAST_CastExprInterface] > {

    let description = [{
        An interface to provide [`clang::ExplicitCastExpr`]
        (https://clang.llvm.org/doxygen/classclang_1_1ExplicitCastExpr.html) functionality.
    }];

    let methods = [];
}

def VAST_CStyleCastExprInterface : VAST_ASTOpInterface< "CStyleCastExprInterface",
    [VAST_ExplicitCastExprInterface] > {

    let description = [{
        An interface to provide [`clang::CStyleCastExpr`]
        (https://clang.llvm.org/doxygen/classclang_1_1CStyleCastExpr.html) functionality.
    }];

    let methods = [];
}

def VAST_OMPArrayShapingExprInterface : VAST_ASTOpInterface< "OMPArrayShapingExprInterface",
    [VAST_ExprInterface] > {

    let description = [{
        An interface to provide [`clang::OMPArrayShapingExpr`]
        (https://clang.llvm.org/doxygen/classclang_1_1OMPArrayShapingExpr.html) functionality.
    }];

    let methods = [];
}

def VAST_OMPArraySectionExprInterface : VAST_ASTOpInterface< "OMPArraySectionExprInterface",
    [VAST_ExprInterface] > {

    let description = [{
    }];

    let methods = [];
}

def VAST_CompoundLiteralExprInterface : VAST_ASTOpInterface< "CompoundLiteralExprInterface",
    [VAST_ExprInterface] > {

    let description = [{
        An interface to provide [`clang::CompoundLiteralExpr`]
        (https://clang.llvm.org/doxygen/classclang_1_1CompoundLiteralExpr.html) functionality.
    }];

    let methods = [];
}

def VAST_ExtVectorElementExprInterface : VAST_ASTOpInterface< "ExtVectorElementExprInterface",
    [VAST_ExprInterface] > {

    let description = [{
        An interface to provide [`clang::ExtVectorElementExpr`]
        (https://clang.llvm.org/doxygen/classclang_1_1ExtVectorElementExpr.html) functionality.
    }];

    let methods = [];
}

def VAST_InitListExprInterface : VAST_ASTOpInterface< "InitListExprInterface",
    [VAST_ExprInterface] > {

    let description = [{
        An interface to provide [`clang::InitListExpr`]
        (https://clang.llvm.org/doxygen/classclang_1_1InitListExpr.html) functionality.
    }];

    let methods = [];
}

def VAST_DesignatedInitExprInterface : VAST_ASTOpInterface< "DesignatedInitExprInterface",
    [VAST_ExprInterface] > {

    let description = [{
        An interface to provide [`clang::DesignatedInitExpr`]
        (https://clang.llvm.org/doxygen/classclang_1_1DesignatedInitExpr.html) functionality.
    }];

    let methods = [];
}

def VAST_DesignatedInitUpdateExprInterface : VAST_ASTOpInterface< "DesignatedInitUpdateExprInterface",
    [VAST_ExprInterface] > {

    let description = [{
        An interface to provide [`clang::DesignatedInitUpdateExpr`]
        (https://clang.llvm.org/doxygen/classclang_1_1DesignatedInitUpdateExpr.html) functionality.
    }];

    let methods = [];
}

def VAST_ImplicitValueInitExprInterface : VAST_ASTOpInterface< "ImplicitValueInitExprInterface",
    [VAST_ExprInterface] > {

    let description = [{
        An interface to provide [`clang::ImplicitValueInitExpr`]
        (https://clang.llvm.org/doxygen/classclang_1_1ImplicitValueInitExpr.html) functionality.
    }];

    let methods = [];
}

def VAST_NoInitExprInterface : VAST_ASTOpInterface< "NoInitExprInterface",
    [VAST_ExprInterface] > {

    let description = [{
        An interface to provide [`clang::NoInitExpr`]
        (https://clang.llvm.org/doxygen/classclang_1_1NoInitExpr.html) functionality.
    }];

    let methods = [];
}

def VAST_ArrayInitLoopExprInterface : VAST_ASTOpInterface< "ArrayInitLoopExprInterface",
    [VAST_ExprInterface] > {

    let description = [{
        An interface to provide [`clang::ArrayInitLoopExpr`]
        (https://clang.llvm.org/doxygen/classclang_1_1ArrayInitLoopExpr.html) functionality.
    }];

    let methods = [];
}

def VAST_ArrayInitIndexExprInterface : VAST_ASTOpInterface< "ArrayInitIndexExprInterface",
    [VAST_ExprInterface] > {

    let description = [{
        An interface to provide [`clang::ArrayInitIndexExpr`]
        (https://clang.llvm.org/doxygen/classclang_1_1ArrayInitIndexExpr.html) functionality.
    }];

    let methods = [];
}

def VAST_ParenListExprInterface : VAST_ASTOpInterface< "ParenListExprInterface",
    [VAST_ExprInterface] > {

    let description = [{
        An interface to provide [`clang::ParenListExpr`]
        (https://clang.llvm.org/doxygen/classclang_1_1ParenListExpr.html) functionality.
    }];

    let methods = [];
}

def VAST_VAArgExprInterface : VAST_ASTOpInterface< "VAArgExprInterface",
    [VAST_ExprInterface] > {

    let description = [{
        An interface to provide [`clang::VAArgExpr`]
        (https://clang.llvm.org/doxygen/classclang_1_1VAArgExpr.html) functionality.
    }];

    let methods = [];
}

def VAST_GenericSelectionExprInterface : VAST_ASTOpInterface< "GenericSelectionExprInterface",
    [VAST_ExprInterface] > {

    let description = [{
        An interface to provide [`clang::GenericSelectionExpr`]
        (https://clang.llvm.org/doxygen/classclang_1_1GenericSelectionExpr.html) functionality.
    }];

    let methods = [];
}

def VAST_PseudoObjectExprInterface : VAST_ASTOpInterface< "PseudoObjectExprInterface",
    [VAST_ExprInterface] > {

    let description = [{
        An interface to provide [`clang::PseudoObjectExpr`]
        (https://clang.llvm.org/doxygen/classclang_1_1PseudoObjectExpr.html) functionality.
    }];

    let methods = [];
}

def VAST_SourceLocExprInterface : VAST_ASTOpInterface< "SourceLocExprInterface",
    [VAST_ExprInterface] > {

    let description = [{
        An interface to provide [`clang::SourceLocExpr`]
        (https://clang.llvm.org/doxygen/classclang_1_1SourceLocExpr.html) functionality.
    }];

    let methods = [];
}

def VAST_FullExprInterface : VAST_ASTOpInterface< "FullExprInterface",
    [VAST_ExprInterface] > {

    let description = [{
        An interface to provide [`clang::FullExpr`]
        (https://clang.llvm.org/doxygen/classclang_1_1FullExpr.html) functionality.
    }];

    let methods = [];
}

def VAST_ConstantExprInterface : VAST_ASTOpInterface< "ConstantExprInterface",
    [VAST_FullExprInterface] > {

    let description = [{
        An interface to provide [`clang::ConstantExpr`]
        (https://clang.llvm.org/doxygen/classclang_1_1ConstantExpr.html) functionality.
    }];

    let methods = [];
}

def VAST_AtomicExprInterface : VAST_ASTOpInterface< "AtomicExprInterface",
    [VAST_ExprInterface] > {

    let description = [{
        An interface to provide [`clang::AtomicExpr`]
        (https://clang.llvm.org/doxygen/classclang_1_1AtomicExpr.html) functionality.
    }];

    let methods = [];
}

def VAST_AddrLabelExprInterface : VAST_ASTOpInterface< "AddrLabelExprInterface",
    [VAST_ExprInterface] > {

    let description = [{
        An interface to provide [`clang::AddrLabelExpr`]
        (https://clang.llvm.org/doxygen/classclang_1_1AddrLabelExpr.html) functionality.
    }];

    let methods = [];
}

def VAST_StmtExprInterface : VAST_ASTOpInterface< "StmtExprInterface",
    [VAST_ExprInterface] > {

    let description = [{
        An interface to provide [`clang::StmtExpr`]
        (https://clang.llvm.org/doxygen/classclang_1_1StmtExpr.html) functionality.
    }];

    let methods = [];
}

def VAST_ChooseExprInterface : VAST_ASTOpInterface< "ChooseExprInterface",
    [VAST_ExprInterface] > {

    let description = [{
        An interface to provide [`clang::ChooseExpr`]
        (https://clang.llvm.org/doxygen/classclang_1_1ChooseExpr.html) functionality.
    }];

    let methods = [];
}

def VAST_GNUNullExprInterface : VAST_ASTOpInterface< "GNUNullExprInterface",
    [VAST_ExprInterface] > {

    let description = [{
        An interface to provide [`clang::GNUNullExpr`]
        (https://clang.llvm.org/doxygen/classclang_1_1GNUNullExpr.html) functionality.
    }];

    let methods = [];
}

def VAST_CXXOperatorCallExprInterface : VAST_ASTOpInterface< "CXXOperatorCallExprInterface",
    [VAST_CallExprInterface] > {

    let description = [{
        An interface to provide [`clang::CXXOperatorCallExpr`]
        (https://clang.llvm.org/doxygen/classclang_1_1CXXOperatorCallExpr.html) functionality.
    }];

    let methods = [];
}

def VAST_CXXMemberCallExprInterface : VAST_ASTOpInterface< "CXXMemberCallExprInterface",
    [VAST_CallExprInterface] > {

    let description = [{
        An interface to provide [`clang::CXXMemberCallExpr`]
        (https://clang.llvm.org/doxygen/classclang_1_1CXXMemberCallExpr.html) functionality.
    }];

    let methods = [];
}

def VAST_CXXRewrittenBinaryOperatorInterface : VAST_ASTOpInterface< "CXXRewrittenBinaryOperatorInterface",
    [VAST_ExprInterface] > {

    let description = [{
        An interface to provide [`clang::CXXRewrittenBinaryOperator`]
        (https://clang.llvm.org/doxygen/classclang_1_1CXXRewrittenBinaryOperator.html) functionality.
    }];

    let methods = [];
}

def VAST_CXXNamedCastExprInterface : VAST_ASTOpInterface< "CXXNamedCastExprInterface",
    [VAST_ExplicitCastExprInterface] > {

    let description = [{
        An interface to provide [`clang::CXXNamedCastExpr`]
        (https://clang.llvm.org/doxygen/classclang_1_1CXXNamedCastExpr.html) functionality.
    }];

    let methods = [];
}

def VAST_CXXStaticCastExprInterface : VAST_ASTOpInterface< "CXXStaticCastExprInterface",
    [VAST_CXXNamedCastExprInterface] > {

    let description = [{
        An interface to provide [`clang::CXXStaticCastExpr`]
        (https://clang.llvm.org/doxygen/classclang_1_1CXXStaticCastExpr.html) functionality.
    }];

    let methods = [];
}

def VAST_CXXDynamicCastExprInterface : VAST_ASTOpInterface< "CXXDynamicCastExprInterface",
    [VAST_CXXNamedCastExprInterface] > {

    let description = [{
        An interface to provide [`clang::CXXDynamicCastExpr`]
        (https://clang.llvm.org/doxygen/classclang_1_1CXXDynamicCastExpr.html) functionality.
    }];

    let methods = [];
}

def VAST_CXXReinterpretCastExprInterface : VAST_ASTOpInterface< "CXXReinterpretCastExprInterface",
    [VAST_CXXNamedCastExprInterface] > {

    let description = [{
        An interface to provide [`clang::CXXReinterpretCastExpr`]
        (https://clang.llvm.org/doxygen/classclang_1_1CXXReinterpretCastExpr.html) functionality.
    }];

    let methods = [];
}

def VAST_CXXConstCastExprInterface : VAST_ASTOpInterface< "CXXConstCastExprInterface",
    [VAST_CXXNamedCastExprInterface] > {

    let description = [{
        An interface to provide [`clang::CXXConstCastExpr`]
        (https://clang.llvm.org/doxygen/classclang_1_1CXXConstCastExpr.html) functionality.
    }];

    let methods = [];
}

def VAST_CXXAddrspaceCastExprInterface : VAST_ASTOpInterface< "CXXAddrspaceCastExprInterface",
    [VAST_CXXNamedCastExprInterface] > {

    let description = [{
        An interface to provide [`clang::CXXAddrspaceCastExpr`]
        (https://clang.llvm.org/doxygen/classclang_1_1CXXAddrspaceCastExpr.html) functionality.
    }];

    let methods = [];
}

def VAST_CXXFunctionalCastExprInterface : VAST_ASTOpInterface< "CXXFunctionalCastExprInterface",
    [VAST_ExplicitCastExprInterface] > {

    let description = [{
        An interface to provide [`clang::CXXFunctionalCastExpr`]
        (https://clang.llvm.org/doxygen/classclang_1_1CXXFunctionalCastExpr.html) functionality.
    }];

    let methods = [];
}

def VAST_CXXTypeidExprInterface : VAST_ASTOpInterface< "CXXTypeidExprInterface",
    [VAST_ExprInterface] > {

    let description = [{
        An interface to provide [`clang::CXXTypeidExpr`]
        (https://clang.llvm.org/doxygen/classclang_1_1CXXTypeidExpr.html) functionality.
    }];

    let methods = [];
}

def VAST_UserDefinedLiteralInterface : VAST_ASTOpInterface< "UserDefinedLiteralInterface",
    [VAST_CallExprInterface] > {

    let description = [{
        An interface to provide [`clang::UserDefinedLiteral`]
        (https://clang.llvm.org/doxygen/classclang_1_1UserDefinedLiteral.html) functionality.
    }];

    let methods = [];
}

def VAST_CXXBoolLiteralExprInterface : VAST_ASTOpInterface< "CXXBoolLiteralExprInterface",
    [VAST_ExprInterface] > {

    let description = [{
        An interface to provide [`clang::CXXBoolLiteralExpr`]
        (https://clang.llvm.org/doxygen/classclang_1_1CXXBoolLiteralExpr.html) functionality.
    }];

    let methods = [];
}

def VAST_CXXNullPtrLiteralExprInterface : VAST_ASTOpInterface< "CXXNullPtrLiteralExprInterface",
    [VAST_ExprInterface] > {

    let description = [{
        An interface to provide [`clang::CXXNullPtrLiteralExpr`]
        (https://clang.llvm.org/doxygen/classclang_1_1CXXNullPtrLiteralExpr.html) functionality.
    }];

    let methods = [];
}

def VAST_CXXThisExprInterface : VAST_ASTOpInterface< "CXXThisExprInterface",
    [VAST_ExprInterface] > {

    let description = [{
        An interface to provide [`clang::CXXThisExpr`]
        (https://clang.llvm.org/doxygen/classclang_1_1CXXThisExpr.html) functionality.
    }];

    let methods = [];
}

def VAST_CXXThrowExprInterface : VAST_ASTOpInterface< "CXXThrowExprInterface",
    [VAST_ExprInterface] > {

    let description = [{
        An interface to provide [`clang::CXXThrowExpr`]
        (https://clang.llvm.org/doxygen/classclang_1_1CXXThrowExpr.html) functionality.
    }];

    let methods = [];
}

def VAST_CXXDefaultArgExprInterface : VAST_ASTOpInterface< "CXXDefaultArgExprInterface",
    [VAST_ExprInterface] > {

    let description = [{
        An interface to provide [`clang::CXXDefaultArgExpr`]
        (https://clang.llvm.org/doxygen/classclang_1_1CXXDefaultArgExpr.html) functionality.
    }];

    let methods = [];
}

def VAST_CXXDefaultInitExprInterface : VAST_ASTOpInterface< "CXXDefaultInitExprInterface",
    [VAST_ExprInterface] > {

    let description = [{
        An interface to provide [`clang::CXXDefaultInitExpr`]
        (https://clang.llvm.org/doxygen/classclang_1_1CXXDefaultInitExpr.html) functionality.
    }];

    let methods = [];
}

def VAST_CXXScalarValueInitExprInterface : VAST_ASTOpInterface< "CXXScalarValueInitExprInterface",
    [VAST_ExprInterface] > {

    let description = [{
        An interface to provide [`clang::CXXScalarValueInitExpr`]
        (https://clang.llvm.org/doxygen/classclang_1_1CXXScalarValueInitExpr.html) functionality.
    }];

    let methods = [];
}

def VAST_CXXStdInitializerListExprInterface : VAST_ASTOpInterface< "CXXStdInitializerListExprInterface",
    [VAST_ExprInterface] > {

    let description = [{
        An interface to provide [`clang::CXXStdInitializerListExpr`]
        (https://clang.llvm.org/doxygen/classclang_1_1CXXStdInitializerListExpr.html) functionality.
    }];

    let methods = [];
}

def VAST_CXXNewExprInterface : VAST_ASTOpInterface< "CXXNewExprInterface",
    [VAST_ExprInterface] > {

    let description = [{
        An interface to provide [`clang::CXXNewExpr`]
        (https://clang.llvm.org/doxygen/classclang_1_1CXXNewExpr.html) functionality.
    }];

    let methods = [];
}

def VAST_CXXDeleteExprInterface : VAST_ASTOpInterface< "CXXDeleteExprInterface",
    [VAST_ExprInterface] > {

    let description = [{
        An interface to provide [`clang::CXXDeleteExpr`]
        (https://clang.llvm.org/doxygen/classclang_1_1CXXDeleteExpr.html) functionality.
    }];

    let methods = [];
}

def VAST_CXXPseudoDestructorExprInterface : VAST_ASTOpInterface< "CXXPseudoDestructorExprInterface",
    [VAST_ExprInterface] > {

    let description = [{
        An interface to provide [`clang::CXXPseudoDestructorExpr`]
        (https://clang.llvm.org/doxygen/classclang_1_1CXXPseudoDestructorExpr.html) functionality.
    }];

    let methods = [];
}

def VAST_TypeTraitExprInterface : VAST_ASTOpInterface< "TypeTraitExprInterface",
    [VAST_ExprInterface] > {

    let description = [{
        An interface to provide [`clang::TypeTraitExpr`]
        (https://clang.llvm.org/doxygen/classclang_1_1TypeTraitExpr.html) functionality.
    }];

    let methods = [];
}

def VAST_ArrayTypeTraitExprInterface : VAST_ASTOpInterface< "ArrayTypeTraitExprInterface",
    [VAST_ExprInterface] > {

    let description = [{
        An interface to provide [`clang::ArrayTypeTraitExpr`]
        (https://clang.llvm.org/doxygen/classclang_1_1ArrayTypeTraitExpr.html) functionality.
    }];

    let methods = [];
}

def VAST_ExpressionTraitExprInterface : VAST_ASTOpInterface< "ExpressionTraitExprInterface",
    [VAST_ExprInterface] > {

    let description = [{
        An interface to provide [`clang::ExpressionTraitExpr`]
        (https://clang.llvm.org/doxygen/classclang_1_1ExpressionTraitExpr.html) functionality.
    }];

    let methods = [];
}

def VAST_DependentScopeDeclRefExprInterface : VAST_ASTOpInterface< "DependentScopeDeclRefExprInterface",
    [VAST_ExprInterface] > {

    let description = [{
        An interface to provide [`clang::DependentScopeDeclRefExpr`]
        (https://clang.llvm.org/doxygen/classclang_1_1DependentScopeDeclRefExpr.html) functionality.
    }];

    let methods = [];
}

def VAST_CXXConstructExprInterface : VAST_ASTOpInterface< "CXXConstructExprInterface",
    [VAST_ExprInterface] > {

    let description = [{
        An interface to provide [`clang::CXXConstructExpr`]
        (https://clang.llvm.org/doxygen/classclang_1_1CXXConstructExpr.html) functionality.
    }];

    let methods = [];
}

def VAST_CXXInheritedCtorInitExprInterface : VAST_ASTOpInterface< "CXXInheritedCtorInitExprInterface",
    [VAST_ExprInterface] > {

    let description = [{
        An interface to provide [`clang::CXXInheritedCtorInitExpr`]
        (https://clang.llvm.org/doxygen/classclang_1_1CXXInheritedCtorInitExpr.html) functionality.
    }];

    let methods = [];
}

def VAST_CXXBindTemporaryExprInterface : VAST_ASTOpInterface< "CXXBindTemporaryExprInterface",
    [VAST_ExprInterface] > {

    let description = [{
        An interface to provide [`clang::CXXBindTemporaryExpr`]
        (https://clang.llvm.org/doxygen/classclang_1_1CXXBindTemporaryExpr.html) functionality.
    }];

    let methods = [];
}

def VAST_ExprWithCleanupsInterface : VAST_ASTOpInterface< "ExprWithCleanupsInterface",
    [VAST_FullExprInterface] > {

    let description = [{
        An interface to provide [`clang::ExprWithCleanups`]
        (https://clang.llvm.org/doxygen/classclang_1_1ExprWithCleanups.html) functionality.
    }];

    let methods = [];
}

def VAST_CXXTemporaryObjectExprInterface : VAST_ASTOpInterface< "CXXTemporaryObjectExprInterface",
    [VAST_CXXConstructExprInterface] > {

    let description = [{
        An interface to provide [`clang::CXXTemporaryObjectExpr`]
        (https://clang.llvm.org/doxygen/classclang_1_1CXXTemporaryObjectExpr.html) functionality.
    }];

    let methods = [];
}

def VAST_CXXUnresolvedConstructExprInterface : VAST_ASTOpInterface< "CXXUnresolvedConstructExprInterface",
    [VAST_ExprInterface] > {

    let description = [{
        An interface to provide [`clang::CXXUnresolvedConstructExpr`]
        (https://clang.llvm.org/doxygen/classclang_1_1CXXUnresolvedConstructExpr.html) functionality.
    }];

    let methods = [];
}

def VAST_CXXDependentScopeMemberExprInterface : VAST_ASTOpInterface< "CXXDependentScopeMemberExprInterface",
    [VAST_ExprInterface] > {

    let description = [{
        An interface to provide [`clang::CXXDependentScopeMemberExpr`]
        (https://clang.llvm.org/doxygen/classclang_1_1CXXDependentScopeMemberExpr.html) functionality.
    }];

    let methods = [];
}

def VAST_OverloadExprInterface : VAST_ASTOpInterface< "OverloadExprInterface",
    [VAST_ExprInterface] > {

    let description = [{
        An interface to provide [`clang::OverloadExpr`]
        (https://clang.llvm.org/doxygen/classclang_1_1OverloadExpr.html) functionality.
    }];

    let methods = [];
}

def VAST_UnresolvedLookupExprInterface : VAST_ASTOpInterface< "UnresolvedLookupExprInterface",
    [VAST_OverloadExprInterface] > {

    let description = [{
        An interface to provide [`clang::UnresolvedLookupExpr`]
        (https://clang.llvm.org/doxygen/classclang_1_1UnresolvedLookupExpr.html) functionality.
    }];

    let methods = [];
}

def VAST_UnresolvedMemberExprInterface : VAST_ASTOpInterface< "UnresolvedMemberExprInterface",
    [VAST_OverloadExprInterface] > {

    let description = [{
        An interface to provide [`clang::UnresolvedMemberExpr`]
        (https://clang.llvm.org/doxygen/classclang_1_1UnresolvedMemberExpr.html) functionality.
    }];

    let methods = [];
}

def VAST_CXXNoexceptExprInterface : VAST_ASTOpInterface< "CXXNoexceptExprInterface",
    [VAST_ExprInterface] > {

    let description = [{
        An interface to provide [`clang::CXXNoexceptExpr`]
        (https://clang.llvm.org/doxygen/classclang_1_1CXXNoexceptExpr.html) functionality.
    }];

    let methods = [];
}

def VAST_PackExpansionExprInterface : VAST_ASTOpInterface< "PackExpansionExprInterface",
    [VAST_ExprInterface] > {

    let description = [{
        An interface to provide [`clang::PackExpansionExpr`]
        (https://clang.llvm.org/doxygen/classclang_1_1PackExpansionExpr.html) functionality.
    }];

    let methods = [];
}

def VAST_SizeOfPackExprInterface : VAST_ASTOpInterface< "SizeOfPackExprInterface",
    [VAST_ExprInterface] > {

    let description = [{
        An interface to provide [`clang::SizeOfPackExpr`]
        (https://clang.llvm.org/doxygen/classclang_1_1SizeOfPackExpr.html) functionality.
    }];

    let methods = [];
}

def VAST_PackIndexingExprInterface : VAST_ASTOpInterface< "PackIndexingExprInterface",
    [VAST_ExprInterface] > {

    let description = [{
        An interface to provide [`clang::PackIndexingExpr`]
        (https://clang.llvm.org/doxygen/classclang_1_1PackIndexingExpr.html) functionality.
    }];

    let methods = [];
}

def VAST_SubstNonTypeTemplateParmExprInterface : VAST_ASTOpInterface< "SubstNonTypeTemplateParmExprInterface",
    [VAST_ExprInterface] > {

    let description = [{
        An interface to provide [`clang::SubstNonTypeTemplateParmExpr`]
        (https://clang.llvm.org/doxygen/classclang_1_1SubstNonTypeTemplateParmExpr.html) functionality.
    }];

    let methods = [];
}

def VAST_SubstNonTypeTemplateParmPackExprInterface : VAST_ASTOpInterface< "SubstNonTypeTemplateParmPackExprInterface",
    [VAST_ExprInterface] > {

    let description = [{
        An interface to provide [`clang::SubstNonTypeTemplateParmPackExpr`]
        (https://clang.llvm.org/doxygen/classclang_1_1SubstNonTypeTemplateParmPackExpr.html) functionality.
    }];

    let methods = [];
}

def VAST_FunctionParmPackExprInterface : VAST_ASTOpInterface< "FunctionParmPackExprInterface",
    [VAST_ExprInterface] > {

    let description = [{
        An interface to provide [`clang::FunctionParmPackExpr`]
        (https://clang.llvm.org/doxygen/classclang_1_1FunctionParmPackExpr.html) functionality.
    }];

    let methods = [];
}

def VAST_MaterializeTemporaryExprInterface : VAST_ASTOpInterface< "MaterializeTemporaryExprInterface",
    [VAST_ExprInterface] > {

    let description = [{
        An interface to provide [`clang::MaterializeTemporaryExpr`]
        (https://clang.llvm.org/doxygen/classclang_1_1MaterializeTemporaryExpr.html) functionality.
    }];

    let methods = [];
}

def VAST_LambdaExprInterface : VAST_ASTOpInterface< "LambdaExprInterface",
    [VAST_ExprInterface] > {

    let description = [{
        An interface to provide [`clang::LambdaExpr`]
        (https://clang.llvm.org/doxygen/classclang_1_1LambdaExpr.html) functionality.
    }];

    let methods = [];
}

def VAST_CXXFoldExprInterface : VAST_ASTOpInterface< "CXXFoldExprInterface",
    [VAST_ExprInterface] > {

    let description = [{
        An interface to provide [`clang::CXXFoldExpr`]
        (https://clang.llvm.org/doxygen/classclang_1_1CXXFoldExpr.html) functionality.
    }];

    let methods = [];
}

def VAST_CXXParenListInitExprInterface : VAST_ASTOpInterface< "CXXParenListInitExprInterface",
    [VAST_ExprInterface] > {

    let description = [{
        An interface to provide [`clang::CXXParenListInitExpr`]
        (https://clang.llvm.org/doxygen/classclang_1_1CXXParenListInitExpr.html) functionality.
    }];

    let methods = [];
}

def VAST_CoroutineSuspendExprInterface : VAST_ASTOpInterface< "CoroutineSuspendExprInterface",
    [VAST_ExprInterface] > {

    let description = [{
        An interface to provide [`clang::CoroutineSuspendExpr`]
        (https://clang.llvm.org/doxygen/classclang_1_1CoroutineSuspendExpr.html) functionality.
    }];

    let methods = [];
}

def VAST_CoawaitExprInterface : VAST_ASTOpInterface< "CoawaitExprInterface",
    [VAST_CoroutineSuspendExprInterface] > {

    let description = [{
        An interface to provide [`clang::CoawaitExpr`]
        (https://clang.llvm.org/doxygen/classclang_1_1CoawaitExpr.html) functionality.
    }];

    let methods = [];
}

def VAST_DependentCoawaitExprInterface : VAST_ASTOpInterface< "DependentCoawaitExprInterface",
    [VAST_ExprInterface] > {

    let description = [{
        An interface to provide [`clang::DependentCoawaitExpr`]
        (https://clang.llvm.org/doxygen/classclang_1_1DependentCoawaitExpr.html) functionality.
    }];

    let methods = [];
}

def VAST_CoyieldExprInterface : VAST_ASTOpInterface< "CoyieldExprInterface",
    [VAST_CoroutineSuspendExprInterface] > {

    let description = [{
        An interface to provide [`clang::CoyieldExpr`]
        (https://clang.llvm.org/doxygen/classclang_1_1CoyieldExpr.html) functionality.
    }];

    let methods = [];
}

def VAST_ConceptSpecializationExprInterface : VAST_ASTOpInterface< "ConceptSpecializationExprInterface",
    [VAST_ExprInterface] > {

    let description = [{
        An interface to provide [`clang::ConceptSpecializationExpr`]
        (https://clang.llvm.org/doxygen/classclang_1_1ConceptSpecializationExpr.html) functionality.
    }];

    let methods = [];
}

def VAST_RequiresExprInterface : VAST_ASTOpInterface< "RequiresExprInterface",
    [VAST_ExprInterface] > {

    let description = [{
        An interface to provide [`clang::RequiresExpr`]
        (https://clang.llvm.org/doxygen/classclang_1_1RequiresExpr.html) functionality.
    }];

    let methods = [];
}

def VAST_ObjCStringLiteralInterface : VAST_ASTOpInterface< "ObjCStringLiteralInterface",
    [VAST_ExprInterface] > {

    let description = [{
        An interface to provide [`clang::ObjCStringLiteral`]
        (https://clang.llvm.org/doxygen/classclang_1_1ObjCStringLiteral.html) functionality.
    }];

    let methods = [];
}

def VAST_ObjCBoxedExprInterface : VAST_ASTOpInterface< "ObjCBoxedExprInterface",
    [VAST_ExprInterface] > {

    let description = [{
        An interface to provide [`clang::ObjCBoxedExpr`]
        (https://clang.llvm.org/doxygen/classclang_1_1ObjCBoxedExpr.html) functionality.
    }];

    let methods = [];
}

def VAST_ObjCArrayLiteralInterface : VAST_ASTOpInterface< "ObjCArrayLiteralInterface",
    [VAST_ExprInterface] > {

    let description = [{
        An interface to provide [`clang::ObjCArrayLiteral`]
        (https://clang.llvm.org/doxygen/classclang_1_1ObjCArrayLiteral.html) functionality.
    }];

    let methods = [];
}

def VAST_ObjCDictionaryLiteralInterface : VAST_ASTOpInterface< "ObjCDictionaryLiteralInterface",
    [VAST_ExprInterface] > {

    let description = [{
        An interface to provide [`clang::ObjCDictionaryLiteral`]
        (https://clang.llvm.org/doxygen/classclang_1_1ObjCDictionaryLiteral.html) functionality.
    }];

    let methods = [];
}

def VAST_ObjCEncodeExprInterface : VAST_ASTOpInterface< "ObjCEncodeExprInterface",
    [VAST_ExprInterface] > {

    let description = [{
        An interface to provide [`clang::ObjCEncodeExpr`]
        (https://clang.llvm.org/doxygen/classclang_1_1ObjCEncodeExpr.html) functionality.
    }];

    let methods = [];
}

def VAST_ObjCMessageExprInterface : VAST_ASTOpInterface< "ObjCMessageExprInterface",
    [VAST_ExprInterface] > {

    let description = [{
        An interface to provide [`clang::ObjCMessageExpr`]
        (https://clang.llvm.org/doxygen/classclang_1_1ObjCMessageExpr.html) functionality.
    }];

    let methods = [];
}

def VAST_ObjCSelectorExprInterface : VAST_ASTOpInterface< "ObjCSelectorExprInterface",
    [VAST_ExprInterface] > {

    let description = [{
        An interface to provide [`clang::ObjCSelectorExpr`]
        (https://clang.llvm.org/doxygen/classclang_1_1ObjCSelectorExpr.html) functionality.
    }];

    let methods = [];
}

def VAST_ObjCProtocolExprInterface : VAST_ASTOpInterface< "ObjCProtocolExprInterface",
    [VAST_ExprInterface] > {

    let description = [{
        An interface to provide [`clang::ObjCProtocolExpr`]
        (https://clang.llvm.org/doxygen/classclang_1_1ObjCProtocolExpr.html) functionality.
    }];

    let methods = [];
}

def VAST_ObjCIvarRefExprInterface : VAST_ASTOpInterface< "ObjCIvarRefExprInterface",
    [VAST_ExprInterface] > {

    let description = [{
        An interface to provide [`clang::ObjCIvarRefExpr`]
        (https://clang.llvm.org/doxygen/classclang_1_1ObjCIvarRefExpr.html) functionality.
    }];

    let methods = [];
}

def VAST_ObjCPropertyRefExprInterface : VAST_ASTOpInterface< "ObjCPropertyRefExprInterface",
    [VAST_ExprInterface] > {

    let description = [{
        An interface to provide [`clang::ObjCPropertyRefExpr`]
        (https://clang.llvm.org/doxygen/classclang_1_1ObjCPropertyRefExpr.html) functionality.
    }];

    let methods = [];
}

def VAST_ObjCIsaExprInterface : VAST_ASTOpInterface< "ObjCIsaExprInterface",
    [VAST_ExprInterface] > {

    let description = [{
        An interface to provide [`clang::ObjCIsaExpr`]
        (https://clang.llvm.org/doxygen/classclang_1_1ObjCIsaExpr.html) functionality.
    }];

    let methods = [];
}

def VAST_ObjCIndirectCopyRestoreExprInterface : VAST_ASTOpInterface< "ObjCIndirectCopyRestoreExprInterface",
    [VAST_ExprInterface] > {

    let description = [{
        An interface to provide [`clang::ObjCIndirectCopyRestoreExpr`]
        (https://clang.llvm.org/doxygen/classclang_1_1ObjCIndirectCopyRestoreExpr.html) functionality.
    }];

    let methods = [];
}

def VAST_ObjCBoolLiteralExprInterface : VAST_ASTOpInterface< "ObjCBoolLiteralExprInterface",
    [VAST_ExprInterface] > {

    let description = [{
        An interface to provide [`clang::ObjCBoolLiteralExpr`]
        (https://clang.llvm.org/doxygen/classclang_1_1ObjCBoolLiteralExpr.html) functionality.
    }];

    let methods = [];
}

def VAST_ObjCSubscriptRefExprInterface : VAST_ASTOpInterface< "ObjCSubscriptRefExprInterface",
    [VAST_ExprInterface] > {

    let description = [{
        An interface to provide [`clang::ObjCSubscriptRefExpr`]
        (https://clang.llvm.org/doxygen/classclang_1_1ObjCSubscriptRefExpr.html) functionality.
    }];

    let methods = [];
}

def VAST_ObjCAvailabilityCheckExprInterface : VAST_ASTOpInterface< "ObjCAvailabilityCheckExprInterface",
    [VAST_ExprInterface] > {

    let description = [{
        An interface to provide [`clang::ObjCAvailabilityCheckExpr`]
        (https://clang.llvm.org/doxygen/classclang_1_1ObjCAvailabilityCheckExpr.html) functionality.
    }];

    let methods = [];
}

def VAST_ObjCBridgedCastExprInterface : VAST_ASTOpInterface< "ObjCBridgedCastExprInterface",
    [VAST_ExplicitCastExprInterface] > {

    let description = [{
        An interface to provide [`clang::ObjCBridgedCastExpr`]
        (https://clang.llvm.org/doxygen/classclang_1_1ObjCBridgedCastExpr.html) functionality.
    }];

    let methods = [];
}

def VAST_CUDAKernelCallExprInterface : VAST_ASTOpInterface< "CUDAKernelCallExprInterface",
    [VAST_CallExprInterface] > {

    let description = [{
        An interface to provide [`clang::CUDAKernelCallExpr`]
        (https://clang.llvm.org/doxygen/classclang_1_1CUDAKernelCallExpr.html) functionality.
    }];

    let methods = [];
}

def VAST_ShuffleVectorExprInterface : VAST_ASTOpInterface< "ShuffleVectorExprInterface",
    [VAST_ExprInterface] > {

    let description = [{
        An interface to provide [`clang::ShuffleVectorExpr`]
        (https://clang.llvm.org/doxygen/classclang_1_1ShuffleVectorExpr.html) functionality.
    }];

    let methods = [];
}

def VAST_ConvertVectorExprInterface : VAST_ASTOpInterface< "ConvertVectorExprInterface",
    [VAST_ExprInterface] > {

    let description = [{
        An interface to provide [`clang::ConvertVectorExpr`]
        (https://clang.llvm.org/doxygen/classclang_1_1ConvertVectorExpr.html) functionality.
    }];

    let methods = [];
}

def VAST_BlockExprInterface : VAST_ASTOpInterface< "BlockExprInterface",
    [VAST_ExprInterface] > {

    let description = [{
        An interface to provide [`clang::BlockExpr`]
        (https://clang.llvm.org/doxygen/classclang_1_1BlockExpr.html) functionality.
    }];

    let methods = [];
}

def VAST_OpaqueValueExprInterface : VAST_ASTOpInterface< "OpaqueValueExprInterface",
    [VAST_ExprInterface] > {

    let description = [{
        An interface to provide [`clang::OpaqueValueExpr`]
        (https://clang.llvm.org/doxygen/classclang_1_1OpaqueValueExpr.html) functionality.
    }];

    let methods = [
        InterfaceMethod< "clang::OpaqueValueExpr", "::vast::ast::ExprInterface", "getSourceExpr" >
    ];
}

def VAST_TypoExprInterface : VAST_ASTOpInterface< "TypoExprInterface",
    [VAST_ExprInterface] > {

    let description = [{
        An interface to provide [`clang::TypoExpr`]
        (https://clang.llvm.org/doxygen/classclang_1_1TypoExpr.html) functionality.
    }];

    let methods = [];
}

def VAST_RecoveryExprInterface : VAST_ASTOpInterface< "RecoveryExprInterface",
    [VAST_ExprInterface] > {

    let description = [{
        An interface to provide [`clang::RecoveryExpr`]
        (https://clang.llvm.org/doxygen/classclang_1_1RecoveryExpr.html) functionality.
    }];

    let methods = [];
}

def VAST_BuiltinBitCastExprInterface : VAST_ASTOpInterface< "BuiltinBitCastExprInterface",
    [VAST_ExplicitCastExprInterface] > {

    let description = [{
        An interface to provide [`clang::BuiltinBitCastExpr`]
        (https://clang.llvm.org/doxygen/classclang_1_1BuiltinBitCastExpr.html) functionality.
    }];

    let methods = [];
}

def VAST_EmbedExprInterface : VAST_ASTOpInterface< "EmbedExprInterface",
    [VAST_ExprInterface] > {

    let description = [{
        An interface to provide [`clang::EmbedExpr`]
        (https://clang.llvm.org/doxygen/classclang_1_1EmbedExpr.html) functionality.
    }];

    let methods = [];
}

def VAST_MSPropertyRefExprInterface : VAST_ASTOpInterface< "MSPropertyRefExprInterface",
    [VAST_ExprInterface] > {

    let description = [{
        An interface to provide [`clang::MSPropertyRefExpr`]
        (https://clang.llvm.org/doxygen/classclang_1_1MSPropertyRefExpr.html) functionality.
    }];

    let methods = [];
}

def VAST_MSPropertySubscriptExprInterface : VAST_ASTOpInterface< "MSPropertySubscriptExprInterface",
    [VAST_ExprInterface] > {

    let description = [{
        An interface to provide [`clang::MSPropertySubscriptExpr`]
        (https://clang.llvm.org/doxygen/classclang_1_1MSPropertySubscriptExpr.html) functionality.
    }];

    let methods = [];
}

def VAST_CXXUuidofExprInterface : VAST_ASTOpInterface< "CXXUuidofExprInterface",
    [VAST_ExprInterface] > {

    let description = [{
        An interface to provide [`clang::CXXUuidofExpr`]
        (https://clang.llvm.org/doxygen/classclang_1_1CXXUuidofExpr.html) functionality.
    }];

    let methods = [];
}

def VAST_AsTypeExprInterface : VAST_ASTOpInterface< "AsTypeExprInterface",
    [VAST_ExprInterface] > {

    let description = [{
        An interface to provide [`clang::AsTypeExpr`]
        (https://clang.llvm.org/doxygen/classclang_1_1AsTypeExpr.html) functionality.
    }];

    let methods = [];
}

#endif // VAST_INTERFACES_AST_EXPR_INTERFACE
