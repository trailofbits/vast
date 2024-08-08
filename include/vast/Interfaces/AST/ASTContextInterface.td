// Copyright (c) 2024-present, Trail of Bits, Inc.

#ifndef VAST_INTERFACES_AST_AST_CONTEXT_INTERFACE
#define VAST_INTERFACES_AST_AST_CONTEXT_INTERFACE

include "mlir/IR/OpBase.td"
include "vast/Interfaces/AST/Common.td"

def VAST_ASTContextInterface : VAST_ASTOpInterface< "ASTContextInterface" > {
    
    let description = [{
        An interface to provide [`clang::ASTContext`]
        (https://clang.llvm.org/doxygen/classclang_1_1ASTContext.html) functionality.
    }];

    let methods = [];
}

#endif // VAST_INTERFACES_AST_AST_CONTEXT_INTERFACE
