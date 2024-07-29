// Copyright (c) 2024-present, Trail of Bits, Inc.

#ifndef VAST_INTERFACES_AST_STMT_INTERFACE
#define VAST_INTERFACES_AST_STMT_INTERFACE

include "mlir/IR/OpBase.td"
include "vast/Interfaces/AST/Common.td"

def VAST_StmtInterface : VAST_ASTOpInterface< "StmtInterface" > {
    
    let description = [{
        An interface to provide [`clang::Stmt`]
        (https://clang.llvm.org/doxygen/classclang_1_1Stmt.html) functionality.
    }];

    let methods = [];
}

def VAST_DeclStmtInterface : VAST_ASTOpInterface< "DeclStmtInterface",
    [VAST_StmtInterface] > {
    
    let description = [{
        And interface to provide [`clang;:DeclStmt`]
        (https://clang.llvm.org/doxygen/classclang_1_1DeclStmt.html) functionality.
    }];

    let methods = [
    ];
}

#endif // VAST_INTERFACES_AST_STMT_INTERFACE
