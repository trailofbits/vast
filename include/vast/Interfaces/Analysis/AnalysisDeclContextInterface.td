// Copyright (c) 2024-present, Trail of Bits, Inc.

#ifndef VAST_INTERFACES_ANALYSIS_ANALYSIS_DECL_CONTEXT_INTERFACE
#define VAST_INTERFACES_ANALYSIS_ANALYSIS_DECL_CONTEXT_INTERFACE

include "mlir/IR/OpBase.td"
include "vast/Interfaces/Analysis/Common.td"

def VAST_AnalysisDeclContextInterface : VAST_AnalysisOpInterface< "AnalysisDeclContextInterface" > {

    let description = [{
        An interface to provide [`clang::AnalysisDeclContext`]
        (https://clang.llvm.org/doxygen/classclang_1_1AnalysisDeclContext.html) functionality.
    }];

    let methods = [
        InterfaceMethod< "clang::AnalysisDeclContext::getDecl", "::vast::ast::DeclInterface", "getDecl" >,
        InterfaceMethod< "clang::AnalysisDeclContext::getCFG", "::vast::cfg::CFGInterface", "getCFG" >,
        InterfaceMethod< "clang::AnalysisDeclContext::getASTContext", "::vast::ast::ASTContextInterface", "getASTContext" >
    ];

    let extraClassDeclaration = [{
        template< typename T >
        T *getAnalysis();
    }];
}

#endif // VAST_INTERFACES_ANALYSIS_ANALYSIS_DECL_INTERFACE
