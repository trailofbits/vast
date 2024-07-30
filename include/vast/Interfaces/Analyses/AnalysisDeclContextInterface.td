// Copyright (c) 2024-present, Trail of Bits, Inc.

#ifndef VAST_INTERFACES_ANALYSES_ANALYSES_DECL_CONTEXT_INTERFACE
#define VAST_INTERFACES_ANALYSES_ANALYSES_DECL_CONTEXT_INTERFACE

include "mlir/IR/OpBase.td"
include "vast/Interfaces/Analyses/Common.td"

def VAST_AnalysisDeclContextInterface : VAST_AnalysesOpInterface< "AnalysisDeclContextInterface" > {

    let description = [{
        An interface to provide [`clang::AnalysisDeclContext`]
        (https://clang.llvm.org/doxygen/classclang_1_1AnalysisDeclContext.html) functionality.
    }];

    let methods = [
        InterfaceMethod< "clang::AnalysisDeclContext::getDecl", "::vast::ast::DeclInterface", "getDecl" >
    ];
}

#endif // VAST_INTERFACES_ANALYSES_ANALYSES_DECL_INTERFACE
