// Copyright (c) 2024-present, Trail of Bits, Inc.

#ifndef VAST_INTERFACES_AST_TYPE_INTERFACE
#define VAST_INTERFACES_AST_TYPE_INTERFACE

include "mlir/IR/OpBase.td"
include "vast/Interfaces/AST/Common.td"

def VAST_TypeInterface : VAST_ASTTypeInterface< "TypeInterface" > {

    let description = [{
        An interface to provide [`clang::Type`]
        (https://clang.llvm.org/doxygen/classclang_1_1Type.html) functionality.
    }];

    let methods = [
        InterfaceMethod< "clang::Type::isScalarType", "bool", "isScalarType", (ins) >,
        InterfaceMethod< "clang::Type::isVectorType", "bool", "isVectorType", (ins) >,
        InterfaceMethod< "clang::Type::isRecordType", "bool", "isRecordType", (ins) >,
        InterfaceMethod< "clang::Type::isVoidType", "bool", "isVoidType", (ins) >,
        InterfaceMethod< "clang::Type::isRVVSizelessBuiltinType", "bool", "isRVVSizelessBuiltinType", (ins) >,
        InterfaceMethod< "clang::Type::isAnyPointerType", "bool", "isAnyPointerType", (ins) >,
        InterfaceMethod< "clang::Type::getPointeeType", "::vast::ast::QualTypeInterface", "getPointeeType" >
    ];
}

def VAST_QualTypeInterface : VAST_ASTTypeInterface< "QualTypeInterface" > {

    let description = [{
        An interface to provide [`clang::QualType`]
        (https://clang.llvm.org/doxygen/classclang_1_1QualType.html) functionality.
    }];
    
    let methods = [
        InterfaceMethod< "clang::QualType::getTypePtr", "::vast::ast::TypeInterface", "getTypePtr", (ins) >,
        InterfaceMethod< "clang::QualType::isConstQualified", "bool", "isConstQualified", (ins) >
    ];

    let extraClassDeclaration = [{
        vast::ast::TypeInterface *operator->();
    }];
}

#endif // VAST_INTERFACES_AST_TYPE_INTERFACE
