// Copyright (c) 2024-present, Trail of Bits, Inc.

#ifndef VAST_INTERFACES_AST_DECL_INTERFACE
#define VAST_INTERFACES_AST_DECL_INTERFACE

include "mlir/IR/OpBase.td"
include "vast/Interfaces/AST/Common.td"

def VAST_TagDeclInterface : VAST_ASTOpInterface< "VAST_TagDeclInterface" > {

    let description = [{
        An interface to provide [`clang::TagDecl`]
        (https://clang.llvm.org/doxygen/classclang_1_1TagDecl.html) functionality.
    }];

    let methods = [];
}

def VAST_DeclContextInterface : VAST_ASTOpInterface< "DeclContextInterface" > {

    let description = [{
        An interface to provide [`clang::DeclContext`]
        (https://clang.llvm.org/doxygen/classclang_1_1DeclContext.html) functionality.
    }];

    let methods = [
        InterfaceMethod< "clang::DeclContext::getParentASTContext",
                         "::vast::ast::ASTContextInterface", "getParentASTContext", (ins) >,
        InterfaceMethod< "clang::DeclContext::decls_begin", "::vast::analyses::decl_interface_iterator", "decls_begin", (ins) >,
        InterfaceMethod< "clang::DeclContext::decls_end", "::vast::analyses::decl_interface_iterator", "decls_end", (ins) >
    ];
}

def VAST_DeclInterface : VAST_ASTOpInterface< "DeclInterface" > {

    let description = [{
        An interface to provide [`clang::Decl`]
        (https://clang.llvm.org/doxygen/classclang_1_1Decl.html) functionality.
    }];

    let methods = [
        InterfaceMethod< "clang::Decl::isImplicit", "bool", "isImplicit", (ins) >,
        InterfaceMethod< "clang::Decl::getDeclContext", "::vast::ast::DeclContextInterface", "getDeclContext", (ins) >,
        InterfaceMethod< "clang::Decl::getASTContext", "::vast::ast::ASTContextInterface", "getASTContext", (ins) >,
        InterfaceMethod< "clang::Decl::getNextDeclInContext", "::vast::ast::DeclInterface *", "getNextDeclInContext", (ins) >
    ];

    // let extraClassDeclaration = [];
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
        InterfaceMethod< "clang::ValueDecl::getType", "::vast::ast::QualTypeInterface", "getType", (ins) >
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
        InterfaceMethod< "clang::VarDecl::getInit", "::vast::ast::ExprInterface", "getInit", (ins) >,
    ];
}

def VAST_RecordDeclInterface : VAST_ASTOpInterface< "RecordDeclInterface",
    [VAST_TagDeclInterface] > {

    let description = [{
        An interface to provide [`clang::RecordDecl`]
        (https://clang.llvm.org/doxygen/classclang_1_1RecordDecl.html) functionality.
    }];

    let methods = [
        InterfaceMethod< "clang::RecordDecl::fields", "std::vector< ::vast::ast::FieldDeclInterface >",
                         "fields", (ins) >
    ];
}

#endif // VAST_INTERFACES_AST_DECL_INTERFACE
