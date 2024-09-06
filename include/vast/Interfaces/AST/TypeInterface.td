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
        InterfaceMethod< "clang::Type::isScalarType", "bool", "isScalarType", (ins),
                         [{}], [{ return false; }] >,
        InterfaceMethod< "clang::Type::isVectorType", "bool", "isVectorType", (ins),
                         [{}], [{ return false; }] >,
        InterfaceMethod< "clang::Type::isRecordType", "bool", "isRecordType", (ins),
                         [{}], [{ return false; }] >,
        InterfaceMethod< "clang::Type::isVoidType", "bool", "isVoidType", (ins),
                         [{}], [{ return false; }] >,
        InterfaceMethod< "clang::Type::isRVVSizelessBuiltinType", "bool", "isRVVSizelessBuiltinType", (ins),
                         [{}], [{ return false; }] >,
        InterfaceMethod< "clang::Type::isAnyPointerType", "bool", "isAnyPointerType", (ins),
                         [{}], [{ return false; }] >,
        InterfaceMethod< "clang::Type::getPointeeType", "::vast::ast::QualTypeInterface *", "getPointeeType", (ins),
                         [{}], [{ return nullptr; }] >
    ];
}

def VAST_QualTypeInterface : VAST_ASTTypeInterface< "QualTypeInterface" > {

    let description = [{
        An interface to provide [`clang::QualType`]
        (https://clang.llvm.org/doxygen/classclang_1_1QualType.html) functionality.
    }];
    
    let methods = [
        InterfaceMethod< "clang::QualType::getTypePtr", "::vast::ast::TypeInterface", "getTypePtr", (ins),
                         [{}], [{ return {}; }] >,
        InterfaceMethod< "clang::QualType::isConstQualified", "bool", "isConstQualified", (ins),
                         [{}], [{ return false; }]  >
    ];

    let extraClassDeclaration = [{
        vast::ast::TypeInterface *operator->();
    }];
}

#endif // VAST_INTERFACES_AST_TYPE_INTERFACE
