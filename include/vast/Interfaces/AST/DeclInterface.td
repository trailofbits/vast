// Copyright (c) 2024-present, Trail of Bits, Inc.

#ifndef VAST_INTERFACES_AST_DECL_INTERFACE
#define VAST_INTERFACES_AST_DECL_INTERFACE

include "mlir/IR/OpBase.td"
include "vast/Interfaces/AST/Common.td"

def VAST_TagDeclInterface : VAST_ASTOpInterface< "TagDeclInterface" > {

    let description = [{
        An interface to provide [`clang::TagDecl`]
        (https://clang.llvm.org/doxygen/classclang_1_1TagDecl.html) functionality.
    }];

    let methods = [];
}

def VAST_DeclInterface : VAST_ASTOpInterface< "DeclInterface" > {

    let description = [{
        An interface to provide [`clang::Decl`]
        (https://clang.llvm.org/doxygen/classclang_1_1Decl.html) functionality.
    }];

    let methods = [
        InterfaceMethod< "clang::Decl::isImplicit", "bool", "isImplicit", (ins) >,
        InterfaceMethod< "clang::Decl::getDeclContext", "void", "getDeclContext", (ins) >,
        InterfaceMethod< "clang::Decl::getASTContext", "void", "getASTContext", (ins) >
    ];
}

def VAST_NamedDeclInterface : VAST_ASTOpInterface< "NamedDeclInterface",
    [VAST_DeclInterface] > {

    let description = [{
        An interface to provide [`clang::NamedDecl`]
        (https://clang.llvm.org/doxygen/classclang_1_1NamedDecl.html) functionality.
    }];

    let methods = [];
}

def VAST_ValueDeclInterface : VAST_ASTOpInterface< "ValueDeclInterface",
    [VAST_NamedDeclInterface] > {

    let description = [{
        An interface to provide [`clang::ValueDecl`]
        (https://clang.llvm.org/doxygen/classclang_1_1ValueDecl.html) functionality.
    }];

    let methods = [
        InterfaceMethod< "clang::ValueDecl::getType", "void", "getType", (ins) >
    ];
}

def VAST_DeclaratorDeclInterface : VAST_ASTOpInterface< "DeclaratorDeclInterface",
    [VAST_ValueDeclInterface] > {

    let description = [{
        An interface to provide [`clang::DeclaratorDecl`]
        (https://clang.llvm.org/doxygen/classclang_1_1DeclaratorDecl.html) functionality.
    }];

    let methods = [];
}

def VAST_FieldDeclInterface : VAST_ASTOpInterface< "FieldDeclInterface",
    [VAST_DeclaratorDeclInterface] > {

    let description = [{
        An interface to provide [`clang::FieldDecl`]
        (https://clang.llvm.org/doxygen/classclang_1_1DeclaratorDecl.html) functionality.
    }];

    let methods = [
        InterfaceMethod< "clang::FieldDecl::isUnnamedBitField", "bool", "isUnnamedBitField", (ins) >,
        InterfaceMethod< "clang::FieldDecl::isZeroSize", "bool", "isZeroSize", (ins) >
    ];
}

def VAST_VarDeclInterface : VAST_ASTOpInterface< "VarDeclInterface",
    [VAST_DeclaratorDeclInterface] > {

    let description = [{
        An interface to provide [`clang::VarDecl`]
        (https://clang.llvm.org/doxygen/classclang_1_1VarDecl.html) functionality.
    }];

    let methods = [
        InterfaceMethod< "clang::VarDecl::isLocalVarDecl", "bool", "isLocalVarDecl", (ins) >,
        InterfaceMethod< "clang::VarDecl::hasGlobalStorage", "bool", "hasGlobalStorage", (ins) >,
        InterfaceMethod< "clang::VarDecl::isExceptionVariable", "bool", "isExceptionVariable", (ins) >,
        InterfaceMethod< "clang::VarDecl::isInitCapture", "bool", "isInitCapture", (ins) >,
        InterfaceMethod< "clang::VarDecl::getInit", "void", "getInit", (ins) >
    ];
}

#endif // VAST_INTERFACES_AST_DECL_INTERFACE
