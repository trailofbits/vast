// Copyright (c) 2024-present, Trail of Bits, Inc.

#ifndef VAST_INTERFACES_CFG_CFG_INTERFACE
#define VAST_INTERFACES_CFG_CFG_INTERFACE

include "mlir/IR/OpBase.td"
include "vast/Interfaces/CFG/Common.td"

def VAST_CFGElementInterface : VAST_CFGOpInterface< "CFGElementInterface" > {
    
    let description = [{
        An interface to provide [`clang::CFGElement`]
        (https://clang.llvm.org/doxygen/classclang_1_1CFGElement.html) functionality.
    }];

    let methods = [];

    let extraClassDeclaration = [{
        // clang::CFGElement::getAs
        template< typename T >
        std::optional< T > getAs();
    }];
}

def VAST_CFGStmtInterface : VAST_CFGOpInterface< "CFGStmtInterface",
    [VAST_CFGElementInterface] > {
    
    let description = [{
        An interface to provide [`clang::CFGStmt`]
        (https://clang.llvm.org/doxygen/classclang_1_1CFGStmt.html) functionality.
    }];

    let methods = [
        InterfaceMethod< "clang:;CFGElement::getStmt", "::vast::ast::StmtInterface", "getStmt" >
    ];
}

def VAST_CFGTerminatorInterface : VAST_CFGOpInterface< "CFGTerminatorInterface" > {

    let description = [{
        An interface to provide [`clang::CFGTerminator`]
        (https://clang.llvm.org/doxygen/classclang_1_1CFGTerminator.html) functionality.
    }];

    let methods = [
        InterfaceMethod< "clang::CFGTerminator::getStmt", "::vast::ast::StmtInterface", "getStmt" >
    ];
}

def VAST_CFGBlockInterface : VAST_CFGOpInterface< "CFGBlockInterface" > {
    
    let description = [{
        An interface to provide [`clang::CFGBlock`]
        (https://clang.llvm.org/doxygen/classclang_1_1CFGBlock.html) functionality.
    }];

    let methods = [
        InterfaceMethod< "clang::CFGBlock::getBlockID", "unsigned", "getBlockID" >,
        InterfaceMethod< "clang::CFGBlock::getLabel", "::vast::ast::StmtInterface", "getLabel" >,
        InterfaceMethod< "clang::CFGBlock::pred_begin", "::vast::cfg::pred_iterator", "pred_begin" >,
        InterfaceMethod< "clang::CFGBlock::pred_end", "::vast::cfg::pred_iterator", "pred_end" >,
        InterfaceMethod< "clang::CFGBlock::succs", "::vast::cfg::succ_range", "succs" >,
        InterfaceMethod< "clang:;CFGBlock::succ_size", "unsigned", "succ_size" >,
        InterfaceMethod< "clang::CFGBlock::succ_begin", "::vast::cfg::succ_iterator", "succ_begin" >,
        InterfaceMethod< "clang::CFGBlock::succ_end", "::vast::cfg::succ_iterator", "succ_end" >,
        InterfaceMethod< "clang::CFGBlock::getTerminatorStmt", "::vast::ast::StmtInterface", "getTerminatorStmt" >,
        InterfaceMethod< "clang::CFGBlock::begin", "::vast::cfg::CFGBlockIterator", "begin" >,
        InterfaceMethod< "clang::CFGBlock::end", "::vast::cfg::CFGBlockIterator", "end" >,
        InterfaceMethod< "clang::CFGBlock::getTerminator", "::vast::cfg::CFGTerminatorInterface", "getTerminator" >
    ];
}

def VAST_CFGInterface : VAST_CFGOpInterface< "CFGInterface" > {

    let description = [{
        An interface to provide [`clang::CFG`]
        (https://clang.llvm.org/doxygen/classclang_1_1CFG.html) functionality.
    }];

    let methods = [
        InterfaceMethod< "clang::CFG::getEntry", "::vast::cfg::CFGBlockInterface", "getEntry" >,
        InterfaceMethod< "clang::CFG::getNumBlockIDs", "unsigned", "getNumBlockIDs" >,
        InterfaceMethod< "clang::CFG::begin", "::vast::cfg::CFGIterator", "begin" >,
        InterfaceMethod< "clang::CFG::end", "::vast::cfg::CFGIterator", "end" >
    ];

    let extraClassDeclaration = [{
        // clang::CFG::VisitBlockStmts
        template< typename Callback >
        void VisitBlockStmts(Callback &);
    }];
}

#endif // VAST_INTERFACES_CFG_CFG_DECL_INTERFACE
