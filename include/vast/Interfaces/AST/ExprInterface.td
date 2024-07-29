// Copyright (c) 2024-present, Trail of Bits, Inc.

#ifndef VAST_INTERFACES_AST_EXPR_INTERFACE
#define VAST_INTERFACES_AST_EXPR_INTERFACE

include "mlir/IR/OpBase.td"
include "vast/Interfaces/AST/Common.td"
include "vast/Interfaces/AST/StmtInterface.td"

def VAST_ExprInterface : VAST_ASTOpInterface< "ExprInterface",
    [VAST_StmtInterface] > {
    
    let description = [{
        An interface to provide [`clang::Expr`]
        (https://clang.llvm.org/doxygen/classclang_1_1Expr.html) functionality.
    }];

    let methods = [
        InterfaceMethod< "clang::Expr::getType", "vast::ast::QualTypeInterface", "getType" >
    ];
}

def VAST_DeclRefExprInterface : VAST_ASTOpInterface< "DeclRefExprInterface",
    [VAST_ExprInterface] > {

    let description = [{
        An interface to provide [`clang::DeclRefExpr`]
        (https://clang.llvm.org/doxygen/classclang_1_1DeclRefExpr.html) functionality.
    }];

    let methods = [];
}

def VAST_CastExprInterface : VAST_ASTOpInterface< "CastExprInterface",
    [VAST_ExprInterface] > {

    let description = [{
        An interface to provide [`clang::CastExpr`]
        (https://clang.llvm.org/doxygen/classclang_1_1CastExpr.html) functionality.
    }];

    let methods = [
        InterfaceMethod< "clang::CastExpr::getCastKind", "clang::CastKind", "getCastKind" >,
        InterfaceMethod< "clang::CastExpr::getSubExpr", "vast::ast::ExprInterface", "getSubExpr" >,
    ];
}

def VAST_ExplicitCastExprInterface : VAST_ASTOpInterface< "ExplicitCastInterface",
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
        (https://clang.llvm.org/doxygen/classclang_1_1ExplicitCastExpr.html) functionality.
    }];

    let methods = [];
}

#endif // VAST_INTERFACES_AST_EXPR_INTERFACE
