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

    let methods = [
        InterfaceMethod< "clang::Stmt::getStmtClass", "::clang::Stmt::StmtClass", "getStmtClass" >
    ];
}

def VAST_NullStmtInterface : VAST_ASTOpInterface< "NullStmtInterface",
    [VAST_StmtInterface] > {

    let description = [{
        An interface to provide [`clang::NullStmt`]
        (https://clang.llvm.org/doxygen/classclang_1_1NullStmt.html) functionality.
    }];

    let methods = [];
}

def VAST_CompoundStmtInterface : VAST_ASTOpInterface< "CompoundStmtInterface",
    [VAST_StmtInterface] > {

    let description = [{
        An interface to provide [`clang::CompoundStmt`]
        (https://clang.llvm.org/doxygen/classclang_1_1CompoundStmt.html) functionality.
    }];

    let methods = [];
}

def VAST_IfStmtInterface : VAST_ASTOpInterface< "IfStmtInterface",
    [VAST_StmtInterface] > {

    let description = [{
        An interface to provide [`clang::IfStmt`]
        (https://clang.llvm.org/doxygen/classclang_1_1IfStmt.html) functionality.
    }];

    let methods = [];
}

def VAST_SwitchStmtInterface : VAST_ASTOpInterface< "SwitchStmtInterface",
    [VAST_StmtInterface] > {

    let description = [{
        An interface to provide [`clang::SwitchStmt`]
        (https://clang.llvm.org/doxygen/classclang_1_1SwitchStmt.html) functionality.
    }];

    let methods = [];
}

def VAST_WhileStmtInterface : VAST_ASTOpInterface< "WhileStmtInterface",
    [VAST_StmtInterface] > {

    let description = [{
        An interface to provide [`clang::WhileStmt`]
        (https://clang.llvm.org/doxygen/classclang_1_1WhileStmt.html) functionality.
    }];

    let methods = [];
}

def VAST_DoStmtInterface : VAST_ASTOpInterface< "DoStmtInterface",
    [VAST_StmtInterface] > {

    let description = [{
        An interface to provide [`clang::DoStmt`]
        (https://clang.llvm.org/doxygen/classclang_1_1DoStmt.html) functionality.
    }];

    let methods = [];
}

def VAST_ForStmtInterface : VAST_ASTOpInterface< "ForStmtInterface",
    [VAST_StmtInterface] > {

    let description = [{
        An interface to provide [`clang::ForStmt`]
        (https://clang.llvm.org/doxygen/classclang_1_1ForStmt.html) functionality.
    }];

    let methods = [];
}

def VAST_GotoStmtInterface : VAST_ASTOpInterface< "GotoStmtInterface",
    [VAST_StmtInterface] > {

    let description = [{
        An interface to provide [`clang::GotoStmt`]
        (https://clang.llvm.org/doxygen/classclang_1_1GotoStmt.html) functionality.
    }];

    let methods = [];
}

def VAST_IndirectGotoStmtInterface : VAST_ASTOpInterface< "IndirectGotoStmtInterface",
    [VAST_StmtInterface] > {

    let description = [{
        An interface to provide [`clang::IndirectGotoStmt`]
        (https://clang.llvm.org/doxygen/classclang_1_1IndirectGotoStmt.html) functionality.
    }];

    let methods = [];
}

def VAST_ContinueStmtInterface : VAST_ASTOpInterface< "ContinueStmtInterface",
    [VAST_StmtInterface] > {

    let description = [{
        An interface to provide [`clang::ContinueStmt`]
        (https://clang.llvm.org/doxygen/classclang_1_1ContinueStmt.html) functionality.
    }];

    let methods = [];
}

def VAST_BreakStmtInterface : VAST_ASTOpInterface< "BreakStmtInterface",
    [VAST_StmtInterface] > {

    let description = [{
        An interface to provide [`clang::BreakStmt`]
        (https://clang.llvm.org/doxygen/classclang_1_1BreakStmt.html) functionality.
    }];

    let methods = [];
}

def VAST_ReturnStmtInterface : VAST_ASTOpInterface< "ReturnStmtInterface",
    [VAST_StmtInterface] > {

    let description = [{
        An interface to provide [`clang::ReturnStmt`]
        (https://clang.llvm.org/doxygen/classclang_1_1ReturnStmt.html) functionality.
    }];

    let methods = [];
}

def VAST_DeclStmtInterface : VAST_ASTOpInterface< "DeclStmtInterface",
    [VAST_StmtInterface] > {

    let description = [{
        An interface to provide [`clang::DeclStmt`]
        (https://clang.llvm.org/doxygen/classclang_1_1DeclStmt.html) functionality.
    }];

    let methods = [];
}

def VAST_SwitchCaseInterface : VAST_ASTOpInterface< "SwitchCaseInterface",
    [VAST_StmtInterface] > {

    let description = [{
        An interface to provide [`clang::SwitchCase`]
        (https://clang.llvm.org/doxygen/classclang_1_1SwitchCase.html) functionality.
    }];

    let methods = [];
}

def VAST_CaseStmtInterface : VAST_ASTOpInterface< "CaseStmtInterface",
    [VAST_SwitchCaseInterface] > {

    let description = [{
        An interface to provide [`clang::CaseStmt`]
        (https://clang.llvm.org/doxygen/classclang_1_1CaseStmt.html) functionality.
    }];

    let methods = [];
}

def VAST_DefaultStmtInterface : VAST_ASTOpInterface< "DefaultStmtInterface",
    [VAST_SwitchCaseInterface] > {

    let description = [{
        An interface to provide [`clang::DefaultStmt`]
        (https://clang.llvm.org/doxygen/classclang_1_1DefaultStmt.html) functionality.
    }];

    let methods = [];
}

def VAST_CapturedStmtInterface : VAST_ASTOpInterface< "CapturedStmtInterface",
    [VAST_StmtInterface] > {

    let description = [{
        An interface to provide [`clang::CapturedStmt`]
        (https://clang.llvm.org/doxygen/classclang_1_1CapturedStmt.html) functionality.
    }];

    let methods = [];
}

def VAST_ValueStmtInterface : VAST_ASTOpInterface< "ValueStmtInterface",
    [VAST_StmtInterface] > {

    let description = [{
        An interface to provide [`clang::ValueStmt`]
        (https://clang.llvm.org/doxygen/classclang_1_1ValueStmt.html) functionality.
    }];

    let methods = [];
}

def VAST_LabelStmtInterface : VAST_ASTOpInterface< "LabelStmtInterface",
    [VAST_ValueStmtInterface] > {

    let description = [{
        An interface to provide [`clang::LabelStmt`]
        (https://clang.llvm.org/doxygen/classclang_1_1LabelStmt.html) functionality.
    }];

    let methods = [];
}

def VAST_AttributedStmtInterface : VAST_ASTOpInterface< "AttributedStmtInterface",
    [VAST_ValueStmtInterface] > {

    let description = [{
        An interface to provide [`clang::AttributedStmt`]
        (https://clang.llvm.org/doxygen/classclang_1_1AttributedStmt.html) functionality.
    }];

    let methods = [];
}

def VAST_AsmStmtInterface : VAST_ASTOpInterface< "AsmStmtInterface",
    [VAST_StmtInterface] > {

    let description = [{
        An interface to provide [`clang::AsmStmt`]
        (https://clang.llvm.org/doxygen/classclang_1_1AsmStmt.html) functionality.
    }];

    let methods = [];
}

def VAST_GCCAsmStmtInterface : VAST_ASTOpInterface< "GCCAsmStmtInterface",
    [VAST_AsmStmtInterface] > {

    let description = [{
        An interface to provide [`clang::GCCAsmStmt`]
        (https://clang.llvm.org/doxygen/classclang_1_1GCCAsmStmt.html) functionality.
    }];

    let methods = [];
}

def VAST_MSAsmStmtInterface : VAST_ASTOpInterface< "MSAsmStmtInterface",
    [VAST_AsmStmtInterface] > {

    let description = [{
        An interface to provide [`clang::MSAsmStmt`]
        (https://clang.llvm.org/doxygen/classclang_1_1MSAsmStmt.html) functionality.
    }];

    let methods = [];
}

def VAST_ObjCAtTryStmtInterface : VAST_ASTOpInterface< "ObjCAtTryStmtInterface",
    [VAST_StmtInterface] > {

    let description = [{
        An interface to provide [`clang::ObjCAtTryStmt`]
        (https://clang.llvm.org/doxygen/classclang_1_1ObjCAtTryStmt.html) functionality.
    }];

    let methods = [];
}

def VAST_ObjCAtCatchStmtInterface : VAST_ASTOpInterface< "ObjCAtCatchStmtInterface",
    [VAST_StmtInterface] > {

    let description = [{
        An interface to provide [`clang::ObjCAtCatchStmt`]
        (https://clang.llvm.org/doxygen/classclang_1_1ObjCAtCatchStmt.html) functionality.
    }];

    let methods = [];
}

def VAST_ObjCAtFinallyStmtInterface : VAST_ASTOpInterface< "ObjCAtFinallyStmtInterface",
    [VAST_StmtInterface] > {

    let description = [{
        An interface to provide [`clang::ObjCAtFinallyStmt`]
        (https://clang.llvm.org/doxygen/classclang_1_1ObjCAtFinallyStmt.html) functionality.
    }];

    let methods = [];
}

def VAST_ObjCAtThrowStmtInterface : VAST_ASTOpInterface< "ObjCAtThrowStmtInterface",
    [VAST_StmtInterface] > {

    let description = [{
        An interface to provide [`clang::ObjCAtThrowStmt`]
        (https://clang.llvm.org/doxygen/classclang_1_1ObjCAtThrowStmt.html) functionality.
    }];

    let methods = [];
}

def VAST_ObjCAtSynchronizedStmtInterface : VAST_ASTOpInterface< "ObjCAtSynchronizedStmtInterface",
    [VAST_StmtInterface] > {

    let description = [{
        An interface to provide [`clang::ObjCAtSynchronizedStmt`]
        (https://clang.llvm.org/doxygen/classclang_1_1ObjCAtSynchronizedStmt.html) functionality.
    }];

    let methods = [];
}

def VAST_ObjCForCollectionStmtInterface : VAST_ASTOpInterface< "ObjCForCollectionStmtInterface",
    [VAST_StmtInterface] > {

    let description = [{
        An interface to provide [`clang::ObjCForCollectionStmt`]
        (https://clang.llvm.org/doxygen/classclang_1_1ObjCForCollectionStmt.html) functionality.
    }];

    let methods = [];
}

def VAST_ObjCAutoreleasePoolStmtInterface : VAST_ASTOpInterface< "ObjCAutoreleasePoolStmtInterface",
    [VAST_StmtInterface] > {

    let description = [{
        An interface to provide [`clang::ObjCAutoreleasePoolStmt`]
        (https://clang.llvm.org/doxygen/classclang_1_1ObjCAutoreleasePoolStmt.html) functionality.
    }];

    let methods = [];
}

def VAST_CXXCatchStmtInterface : VAST_ASTOpInterface< "CXXCatchStmtInterface",
    [VAST_StmtInterface] > {

    let description = [{
        An interface to provide [`clang::CXXCatchStmt`]
        (https://clang.llvm.org/doxygen/classclang_1_1CXXCatchStmt.html) functionality.
    }];

    let methods = [];
}

def VAST_CXXTryStmtInterface : VAST_ASTOpInterface< "CXXTryStmtInterface",
    [VAST_StmtInterface] > {

    let description = [{
        An interface to provide [`clang::CXXTryStmt`]
        (https://clang.llvm.org/doxygen/classclang_1_1CXXTryStmt.html) functionality.
    }];

    let methods = [];
}

def VAST_CXXForRangeStmtInterface : VAST_ASTOpInterface< "CXXForRangeStmtInterface",
    [VAST_StmtInterface] > {

    let description = [{
        An interface to provide [`clang::CXXForRangeStmt`]
        (https://clang.llvm.org/doxygen/classclang_1_1CXXForRangeStmt.html) functionality.
    }];

    let methods = [];
}

def VAST_CoroutineBodyStmtInterface : VAST_ASTOpInterface< "CoroutineBodyStmtInterface",
    [VAST_StmtInterface] > {

    let description = [{
        An interface to provide [`clang::CoroutineBodyStmt`]
        (https://clang.llvm.org/doxygen/classclang_1_1CoroutineBodyStmt.html) functionality.
    }];

    let methods = [];
}

def VAST_CoreturnStmtInterface : VAST_ASTOpInterface< "CoreturnStmtInterface",
    [VAST_StmtInterface] > {

    let description = [{
        An interface to provide [`clang::CoreturnStmt`]
        (https://clang.llvm.org/doxygen/classclang_1_1CoreturnStmt.html) functionality.
    }];

    let methods = [];
}

def VAST_SEHTryStmtInterface : VAST_ASTOpInterface< "SEHTryStmtInterface",
    [VAST_StmtInterface] > {

    let description = [{
        An interface to provide [`clang::SEHTryStmt`]
        (https://clang.llvm.org/doxygen/classclang_1_1SEHTryStmt.html) functionality.
    }];

    let methods = [];
}

def VAST_SEHExceptStmtInterface : VAST_ASTOpInterface< "SEHExceptStmtInterface",
    [VAST_StmtInterface] > {

    let description = [{
        An interface to provide [`clang::SEHExceptStmt`]
        (https://clang.llvm.org/doxygen/classclang_1_1SEHExceptStmt.html) functionality.
    }];

    let methods = [];
}

def VAST_SEHFinallyStmtInterface : VAST_ASTOpInterface< "SEHFinallyStmtInterface",
    [VAST_StmtInterface] > {

    let description = [{
        An interface to provide [`clang::SEHFinallyStmt`]
        (https://clang.llvm.org/doxygen/classclang_1_1SEHFinallyStmt.html) functionality.
    }];

    let methods = [];
}

def VAST_SEHLeaveStmtInterface : VAST_ASTOpInterface< "SEHLeaveStmtInterface",
    [VAST_StmtInterface] > {

    let description = [{
        An interface to provide [`clang::SEHLeaveStmt`]
        (https://clang.llvm.org/doxygen/classclang_1_1SEHLeaveStmt.html) functionality.
    }];

    let methods = [];
}

def VAST_MSDependentExistsStmtInterface : VAST_ASTOpInterface< "MSDependentExistsStmtInterface",
    [VAST_StmtInterface] > {

    let description = [{
        An interface to provide [`clang::MSDependentExistsStmt`]
        (https://clang.llvm.org/doxygen/classclang_1_1MSDependentExistsStmt.html) functionality.
    }];

    let methods = [];
}

def VAST_OMPCanonicalLoopInterface : VAST_ASTOpInterface< "OMPCanonicalLoopInterface",
    [VAST_StmtInterface] > {

    let description = [{
        An interface to provide [`clang::OMPCanonicalLoop`]
        (https://clang.llvm.org/doxygen/classclang_1_1OMPCanonicalLoop.html) functionality.
    }];

    let methods = [];
}

def VAST_OMPExecutableDirectiveInterface : VAST_ASTOpInterface< "OMPExecutableDirectiveInterface",
    [VAST_StmtInterface] > {

    let description = [{
        An interface to provide [`clang::OMPExecutableDirective`]
        (https://clang.llvm.org/doxygen/classclang_1_1OMPExecutableDirective.html) functionality.
    }];

    let methods = [];
}

def VAST_OMPMetaDirectiveInterface : VAST_ASTOpInterface< "OMPMetaDirectiveInterface",
    [VAST_OMPExecutableDirectiveInterface] > {

    let description = [{
        An interface to provide [`clang::OMPMetaDirective`]
        (https://clang.llvm.org/doxygen/classclang_1_1OMPMetaDirective.html) functionality.
    }];

    let methods = [];
}

def VAST_OMPLoopBasedDirectiveInterface : VAST_ASTOpInterface< "OMPLoopBasedDirectiveInterface",
    [VAST_OMPExecutableDirectiveInterface] > {

    let description = [{
        An interface to provide [`clang::OMPLoopBasedDirective`]
        (https://clang.llvm.org/doxygen/classclang_1_1OMPLoopBasedDirective.html) functionality.
    }];

    let methods = [];
}

def VAST_OMPLoopDirectiveInterface : VAST_ASTOpInterface< "OMPLoopDirectiveInterface",
    [VAST_OMPLoopBasedDirectiveInterface] > {

    let description = [{
        An interface to provide [`clang::OMPLoopDirective`]
        (https://clang.llvm.org/doxygen/classclang_1_1OMPLoopDirective.html) functionality.
    }];

    let methods = [];
}

def VAST_OMPParallelDirectiveInterface : VAST_ASTOpInterface< "OMPParallelDirectiveInterface",
    [VAST_OMPExecutableDirectiveInterface] > {

    let description = [{
        An interface to provide [`clang::OMPParallelDirective`]
        (https://clang.llvm.org/doxygen/classclang_1_1OMPParallelDirective.html) functionality.
    }];

    let methods = [];
}

def VAST_OMPSimdDirectiveInterface : VAST_ASTOpInterface< "OMPSimdDirectiveInterface",
    [VAST_OMPLoopDirectiveInterface] > {

    let description = [{
        An interface to provide [`clang::OMPSimdDirective`]
        (https://clang.llvm.org/doxygen/classclang_1_1OMPSimdDirective.html) functionality.
    }];

    let methods = [];
}

def VAST_OMPLoopTransformationDirectiveInterface : VAST_ASTOpInterface< "OMPLoopTransformationDirectiveInterface",
    [VAST_OMPLoopBasedDirectiveInterface] > {

    let description = [{
        An interface to provide [`clang::OMPLoopTransformationDirective`]
        (https://clang.llvm.org/doxygen/classclang_1_1OMPLoopTransformationDirective.html) functionality.
    }];

    let methods = [];
}

def VAST_OMPTileDirectiveInterface : VAST_ASTOpInterface< "OMPTileDirectiveInterface",
    [VAST_OMPLoopTransformationDirectiveInterface] > {

    let description = [{
        An interface to provide [`clang::OMPTileDirective`]
        (https://clang.llvm.org/doxygen/classclang_1_1OMPTileDirective.html) functionality.
    }];

    let methods = [];
}

def VAST_OMPUnrollDirectiveInterface : VAST_ASTOpInterface< "OMPUnrollDirectiveInterface",
    [VAST_OMPLoopTransformationDirectiveInterface] > {

    let description = [{
        An interface to provide [`clang::OMPUnrollDirective`]
        (https://clang.llvm.org/doxygen/classclang_1_1OMPUnrollDirective.html) functionality.
    }];

    let methods = [];
}

def VAST_OMPForDirectiveInterface : VAST_ASTOpInterface< "OMPForDirectiveInterface",
    [VAST_OMPLoopDirectiveInterface] > {

    let description = [{
        An interface to provide [`clang::OMPForDirective`]
        (https://clang.llvm.org/doxygen/classclang_1_1OMPForDirective.html) functionality.
    }];

    let methods = [];
}

def VAST_OMPForSimdDirectiveInterface : VAST_ASTOpInterface< "OMPForSimdDirectiveInterface",
    [VAST_OMPLoopDirectiveInterface] > {

    let description = [{
        An interface to provide [`clang::OMPForSimdDirective`]
        (https://clang.llvm.org/doxygen/classclang_1_1OMPForSimdDirective.html) functionality.
    }];

    let methods = [];
}

def VAST_OMPSectionsDirectiveInterface : VAST_ASTOpInterface< "OMPSectionsDirectiveInterface",
    [VAST_OMPExecutableDirectiveInterface] > {

    let description = [{
        An interface to provide [`clang::OMPSectionsDirective`]
        (https://clang.llvm.org/doxygen/classclang_1_1OMPSectionsDirective.html) functionality.
    }];

    let methods = [];
}

def VAST_OMPSectionDirectiveInterface : VAST_ASTOpInterface< "OMPSectionDirectiveInterface",
    [VAST_OMPExecutableDirectiveInterface] > {

    let description = [{
        An interface to provide [`clang::OMPSectionDirective`]
        (https://clang.llvm.org/doxygen/classclang_1_1OMPSectionDirective.html) functionality.
    }];

    let methods = [];
}

def VAST_OMPSingleDirectiveInterface : VAST_ASTOpInterface< "OMPSingleDirectiveInterface",
    [VAST_OMPExecutableDirectiveInterface] > {

    let description = [{
        An interface to provide [`clang::OMPSingleDirective`]
        (https://clang.llvm.org/doxygen/classclang_1_1OMPSingleDirective.html) functionality.
    }];

    let methods = [];
}

def VAST_OMPMasterDirectiveInterface : VAST_ASTOpInterface< "OMPMasterDirectiveInterface",
    [VAST_OMPExecutableDirectiveInterface] > {

    let description = [{
        An interface to provide [`clang::OMPMasterDirective`]
        (https://clang.llvm.org/doxygen/classclang_1_1OMPMasterDirective.html) functionality.
    }];

    let methods = [];
}

def VAST_OMPCriticalDirectiveInterface : VAST_ASTOpInterface< "OMPCriticalDirectiveInterface",
    [VAST_OMPExecutableDirectiveInterface] > {

    let description = [{
        An interface to provide [`clang::OMPCriticalDirective`]
        (https://clang.llvm.org/doxygen/classclang_1_1OMPCriticalDirective.html) functionality.
    }];

    let methods = [];
}

def VAST_OMPParallelForDirectiveInterface : VAST_ASTOpInterface< "OMPParallelForDirectiveInterface",
    [VAST_OMPLoopDirectiveInterface] > {

    let description = [{
        An interface to provide [`clang::OMPParallelForDirective`]
        (https://clang.llvm.org/doxygen/classclang_1_1OMPParallelForDirective.html) functionality.
    }];

    let methods = [];
}

def VAST_OMPParallelForSimdDirectiveInterface : VAST_ASTOpInterface< "OMPParallelForSimdDirectiveInterface",
    [VAST_OMPLoopDirectiveInterface] > {

    let description = [{
        An interface to provide [`clang::OMPParallelForSimdDirective`]
        (https://clang.llvm.org/doxygen/classclang_1_1OMPParallelForSimdDirective.html) functionality.
    }];

    let methods = [];
}

def VAST_OMPParallelMasterDirectiveInterface : VAST_ASTOpInterface< "OMPParallelMasterDirectiveInterface",
    [VAST_OMPExecutableDirectiveInterface] > {

    let description = [{
        An interface to provide [`clang::OMPParallelMasterDirective`]
        (https://clang.llvm.org/doxygen/classclang_1_1OMPParallelMasterDirective.html) functionality.
    }];

    let methods = [];
}

def VAST_OMPParallelSectionsDirectiveInterface : VAST_ASTOpInterface< "OMPParallelSectionsDirectiveInterface",
    [VAST_OMPExecutableDirectiveInterface] > {

    let description = [{
        An interface to provide [`clang::OMPParallelSectionsDirective`]
        (https://clang.llvm.org/doxygen/classclang_1_1OMPParallelSectionsDirective.html) functionality.
    }];

    let methods = [];
}

def VAST_OMPTaskDirectiveInterface : VAST_ASTOpInterface< "OMPTaskDirectiveInterface",
    [VAST_OMPExecutableDirectiveInterface] > {

    let description = [{
        An interface to provide [`clang::OMPTaskDirective`]
        (https://clang.llvm.org/doxygen/classclang_1_1OMPTaskDirective.html) functionality.
    }];

    let methods = [];
}

def VAST_OMPTaskyieldDirectiveInterface : VAST_ASTOpInterface< "OMPTaskyieldDirectiveInterface",
    [VAST_OMPExecutableDirectiveInterface] > {

    let description = [{
        An interface to provide [`clang::OMPTaskyieldDirective`]
        (https://clang.llvm.org/doxygen/classclang_1_1OMPTaskyieldDirective.html) functionality.
    }];

    let methods = [];
}

def VAST_OMPBarrierDirectiveInterface : VAST_ASTOpInterface< "OMPBarrierDirectiveInterface",
    [VAST_OMPExecutableDirectiveInterface] > {

    let description = [{
        An interface to provide [`clang::OMPBarrierDirective`]
        (https://clang.llvm.org/doxygen/classclang_1_1OMPBarrierDirective.html) functionality.
    }];

    let methods = [];
}

def VAST_OMPTaskwaitDirectiveInterface : VAST_ASTOpInterface< "OMPTaskwaitDirectiveInterface",
    [VAST_OMPExecutableDirectiveInterface] > {

    let description = [{
        An interface to provide [`clang::OMPTaskwaitDirective`]
        (https://clang.llvm.org/doxygen/classclang_1_1OMPTaskwaitDirective.html) functionality.
    }];

    let methods = [];
}

def VAST_OMPTaskgroupDirectiveInterface : VAST_ASTOpInterface< "OMPTaskgroupDirectiveInterface",
    [VAST_OMPExecutableDirectiveInterface] > {

    let description = [{
        An interface to provide [`clang::OMPTaskgroupDirective`]
        (https://clang.llvm.org/doxygen/classclang_1_1OMPTaskgroupDirective.html) functionality.
    }];

    let methods = [];
}

def VAST_OMPFlushDirectiveInterface : VAST_ASTOpInterface< "OMPFlushDirectiveInterface",
    [VAST_OMPExecutableDirectiveInterface] > {

    let description = [{
        An interface to provide [`clang::OMPFlushDirective`]
        (https://clang.llvm.org/doxygen/classclang_1_1OMPFlushDirective.html) functionality.
    }];

    let methods = [];
}

def VAST_OMPDepobjDirectiveInterface : VAST_ASTOpInterface< "OMPDepobjDirectiveInterface",
    [VAST_OMPExecutableDirectiveInterface] > {

    let description = [{
        An interface to provide [`clang::OMPDepobjDirective`]
        (https://clang.llvm.org/doxygen/classclang_1_1OMPDepobjDirective.html) functionality.
    }];

    let methods = [];
}

def VAST_OMPScanDirectiveInterface : VAST_ASTOpInterface< "OMPScanDirectiveInterface",
    [VAST_OMPExecutableDirectiveInterface] > {

    let description = [{
        An interface to provide [`clang::OMPScanDirective`]
        (https://clang.llvm.org/doxygen/classclang_1_1OMPScanDirective.html) functionality.
    }];

    let methods = [];
}

def VAST_OMPOrderedDirectiveInterface : VAST_ASTOpInterface< "OMPOrderedDirectiveInterface",
    [VAST_OMPExecutableDirectiveInterface] > {

    let description = [{
        An interface to provide [`clang::OMPOrderedDirective`]
        (https://clang.llvm.org/doxygen/classclang_1_1OMPOrderedDirective.html) functionality.
    }];

    let methods = [];
}

def VAST_OMPAtomicDirectiveInterface : VAST_ASTOpInterface< "OMPAtomicDirectiveInterface",
    [VAST_OMPExecutableDirectiveInterface] > {

    let description = [{
        An interface to provide [`clang::OMPAtomicDirective`]
        (https://clang.llvm.org/doxygen/classclang_1_1OMPAtomicDirective.html) functionality.
    }];

    let methods = [];
}

def VAST_OMPTargetDirectiveInterface : VAST_ASTOpInterface< "OMPTargetDirectiveInterface",
    [VAST_OMPExecutableDirectiveInterface] > {

    let description = [{
        An interface to provide [`clang::OMPTargetDirective`]
        (https://clang.llvm.org/doxygen/classclang_1_1OMPTargetDirective.html) functionality.
    }];

    let methods = [];
}

def VAST_OMPTargetDataDirectiveInterface : VAST_ASTOpInterface< "OMPTargetDataDirectiveInterface",
    [VAST_OMPExecutableDirectiveInterface] > {

    let description = [{
        An interface to provide [`clang::OMPTargetDataDirective`]
        (https://clang.llvm.org/doxygen/classclang_1_1OMPTargetDataDirective.html) functionality.
    }];

    let methods = [];
}

def VAST_OMPTargetEnterDataDirectiveInterface : VAST_ASTOpInterface< "OMPTargetEnterDataDirectiveInterface",
    [VAST_OMPExecutableDirectiveInterface] > {

    let description = [{
        An interface to provide [`clang::OMPTargetEnterDataDirective`]
        (https://clang.llvm.org/doxygen/classclang_1_1OMPTargetEnterDataDirective.html) functionality.
    }];

    let methods = [];
}

def VAST_OMPTargetExitDataDirectiveInterface : VAST_ASTOpInterface< "OMPTargetExitDataDirectiveInterface",
    [VAST_OMPExecutableDirectiveInterface] > {

    let description = [{
        An interface to provide [`clang::OMPTargetExitDataDirective`]
        (https://clang.llvm.org/doxygen/classclang_1_1OMPTargetExitDataDirective.html) functionality.
    }];

    let methods = [];
}

def VAST_OMPTargetParallelDirectiveInterface : VAST_ASTOpInterface< "OMPTargetParallelDirectiveInterface",
    [VAST_OMPExecutableDirectiveInterface] > {

    let description = [{
        An interface to provide [`clang::OMPTargetParallelDirective`]
        (https://clang.llvm.org/doxygen/classclang_1_1OMPTargetParallelDirective.html) functionality.
    }];

    let methods = [];
}

def VAST_OMPTargetParallelForDirectiveInterface : VAST_ASTOpInterface< "OMPTargetParallelForDirectiveInterface",
    [VAST_OMPExecutableDirectiveInterface] > {

    let description = [{
        An interface to provide [`clang::OMPTargetParallelForDirective`]
        (https://clang.llvm.org/doxygen/classclang_1_1OMPTargetParallelForDirective.html) functionality.
    }];

    let methods = [];
}

def VAST_OMPTargetUpdateDirectiveInterface : VAST_ASTOpInterface< "OMPTargetUpdateDirectiveInterface",
    [VAST_OMPExecutableDirectiveInterface] > {

    let description = [{
        An interface to provide [`clang::OMPTargetUpdateDirective`]
        (https://clang.llvm.org/doxygen/classclang_1_1OMPTargetUpdateDirective.html) functionality.
    }];

    let methods = [];
}

def VAST_OMPTeamsDirectiveInterface : VAST_ASTOpInterface< "OMPTeamsDirectiveInterface",
    [VAST_OMPExecutableDirectiveInterface] > {

    let description = [{
        An interface to provide [`clang::OMPTeamsDirective`]
        (https://clang.llvm.org/doxygen/classclang_1_1OMPTeamsDirective.html) functionality.
    }];

    let methods = [];
}

def VAST_OMPCancellationPointDirectiveInterface : VAST_ASTOpInterface< "OMPCancellationPointDirectiveInterface",
    [VAST_OMPExecutableDirectiveInterface] > {

    let description = [{
        An interface to provide [`clang::OMPCancellationPointDirective`]
        (https://clang.llvm.org/doxygen/classclang_1_1OMPCancellationPointDirective.html) functionality.
    }];

    let methods = [];
}

def VAST_OMPCancelDirectiveInterface : VAST_ASTOpInterface< "OMPCancelDirectiveInterface",
    [VAST_OMPExecutableDirectiveInterface] > {

    let description = [{
        An interface to provide [`clang::OMPCancelDirective`]
        (https://clang.llvm.org/doxygen/classclang_1_1OMPCancelDirective.html) functionality.
    }];

    let methods = [];
}

def VAST_OMPScopeDirectiveInterface : VAST_ASTOpInterface< "OMPScopeDirectiveInterface",
    [VAST_OMPExecutableDirectiveInterface] > {

    let description = [{
        An interface to provide [`clang::OMPScopeDirective`]
        (https://clang.llvm.org/doxygen/classclang_1_1OMPScopeDirective.html) functionality.
    }];

    let methods = [];
}

def VAST_OMPTaskLoopDirectiveInterface : VAST_ASTOpInterface< "OMPTaskLoopDirectiveInterface",
    [VAST_OMPLoopDirectiveInterface] > {

    let description = [{
        An interface to provide [`clang::OMPTaskLoopDirective`]
        (https://clang.llvm.org/doxygen/classclang_1_1OMPTaskLoopDirective.html) functionality.
    }];

    let methods = [];
}

def VAST_OMPTaskLoopSimdDirectiveInterface : VAST_ASTOpInterface< "OMPTaskLoopSimdDirectiveInterface",
    [VAST_OMPLoopDirectiveInterface] > {

    let description = [{
        An interface to provide [`clang::OMPTaskLoopSimdDirective`]
        (https://clang.llvm.org/doxygen/classclang_1_1OMPTaskLoopSimdDirective.html) functionality.
    }];

    let methods = [];
}

def VAST_OMPMasterTaskLoopDirectiveInterface : VAST_ASTOpInterface< "OMPMasterTaskLoopDirectiveInterface",
    [VAST_OMPLoopDirectiveInterface] > {

    let description = [{
        An interface to provide [`clang::OMPMasterTaskLoopDirective`]
        (https://clang.llvm.org/doxygen/classclang_1_1OMPMasterTaskLoopDirective.html) functionality.
    }];

    let methods = [];
}

def VAST_OMPMasterTaskLoopSimdDirectiveInterface : VAST_ASTOpInterface< "OMPMasterTaskLoopSimdDirectiveInterface",
    [VAST_OMPLoopDirectiveInterface] > {

    let description = [{
        An interface to provide [`clang::OMPMasterTaskLoopSimdDirective`]
        (https://clang.llvm.org/doxygen/classclang_1_1OMPMasterTaskLoopSimdDirective.html) functionality.
    }];

    let methods = [];
}

def VAST_OMPParallelMasterTaskLoopDirectiveInterface : VAST_ASTOpInterface< "OMPParallelMasterTaskLoopDirectiveInterface",
    [VAST_OMPLoopDirectiveInterface] > {

    let description = [{
        An interface to provide [`clang::OMPParallelMasterTaskLoopDirective`]
        (https://clang.llvm.org/doxygen/classclang_1_1OMPParallelMasterTaskLoopDirective.html) functionality.
    }];

    let methods = [];
}

def VAST_OMPParallelMasterTaskLoopSimdDirectiveInterface : VAST_ASTOpInterface< "OMPParallelMasterTaskLoopSimdDirectiveInterface",
    [VAST_OMPLoopDirectiveInterface] > {

    let description = [{
        An interface to provide [`clang::OMPParallelMasterTaskLoopSimdDirective`]
        (https://clang.llvm.org/doxygen/classclang_1_1OMPParallelMasterTaskLoopSimdDirective.html) functionality.
    }];

    let methods = [];
}

def VAST_OMPMaskedTaskLoopDirectiveInterface : VAST_ASTOpInterface< "OMPMaskedTaskLoopDirectiveInterface",
    [VAST_OMPLoopDirectiveInterface] > {

    let description = [{
        An interface to provide [`clang::OMPMaskedTaskLoopDirective`]
        (https://clang.llvm.org/doxygen/classclang_1_1OMPMaskedTaskLoopDirective.html) functionality.
    }];

    let methods = [];
}

def VAST_OMPMaskedTaskLoopSimdDirectiveInterface : VAST_ASTOpInterface< "OMPMaskedTaskLoopSimdDirectiveInterface",
    [VAST_OMPLoopDirectiveInterface] > {

    let description = [{
        An interface to provide [`clang::OMPMaskedTaskLoopSimdDirective`]
        (https://clang.llvm.org/doxygen/classclang_1_1OMPMaskedTaskLoopSimdDirective.html) functionality.
    }];

    let methods = [];
}

def VAST_OMPParallelMaskedTaskLoopDirectiveInterface : VAST_ASTOpInterface< "OMPParallelMaskedTaskLoopDirectiveInterface",
    [VAST_OMPLoopDirectiveInterface] > {

    let description = [{
        An interface to provide [`clang::OMPParallelMaskedTaskLoopDirective`]
        (https://clang.llvm.org/doxygen/classclang_1_1OMPParallelMaskedTaskLoopDirective.html) functionality.
    }];

    let methods = [];
}

def VAST_OMPParallelMaskedTaskLoopSimdDirectiveInterface : VAST_ASTOpInterface< "OMPParallelMaskedTaskLoopSimdDirectiveInterface",
    [VAST_OMPLoopDirectiveInterface] > {

    let description = [{
        An interface to provide [`clang::OMPParallelMaskedTaskLoopSimdDirective`]
        (https://clang.llvm.org/doxygen/classclang_1_1OMPParallelMaskedTaskLoopSimdDirective.html) functionality.
    }];

    let methods = [];
}

def VAST_OMPDistributeDirectiveInterface : VAST_ASTOpInterface< "OMPDistributeDirectiveInterface",
    [VAST_OMPLoopDirectiveInterface] > {

    let description = [{
        An interface to provide [`clang::OMPDistributeDirective`]
        (https://clang.llvm.org/doxygen/classclang_1_1OMPDistributeDirective.html) functionality.
    }];

    let methods = [];
}

def VAST_OMPDistributeParallelForDirectiveInterface : VAST_ASTOpInterface< "OMPDistributeParallelForDirectiveInterface",
    [VAST_OMPLoopDirectiveInterface] > {

    let description = [{
        An interface to provide [`clang::OMPDistributeParallelForDirective`]
        (https://clang.llvm.org/doxygen/classclang_1_1OMPDistributeParallelForDirective.html) functionality.
    }];

    let methods = [];
}

def VAST_OMPDistributeParallelForSimdDirectiveInterface : VAST_ASTOpInterface< "OMPDistributeParallelForSimdDirectiveInterface",
    [VAST_OMPLoopDirectiveInterface] > {

    let description = [{
        An interface to provide [`clang::OMPDistributeParallelForSimdDirective`]
        (https://clang.llvm.org/doxygen/classclang_1_1OMPDistributeParallelForSimdDirective.html) functionality.
    }];

    let methods = [];
}

def VAST_OMPDistributeSimdDirectiveInterface : VAST_ASTOpInterface< "OMPDistributeSimdDirectiveInterface",
    [VAST_OMPLoopDirectiveInterface] > {

    let description = [{
        An interface to provide [`clang::OMPDistributeSimdDirective`]
        (https://clang.llvm.org/doxygen/classclang_1_1OMPDistributeSimdDirective.html) functionality.
    }];

    let methods = [];
}

def VAST_OMPTargetParallelForSimdDirectiveInterface : VAST_ASTOpInterface< "OMPTargetParallelForSimdDirectiveInterface",
    [VAST_OMPLoopDirectiveInterface] > {

    let description = [{
        An interface to provide [`clang::OMPTargetParallelForSimdDirective`]
        (https://clang.llvm.org/doxygen/classclang_1_1OMPTargetParallelForSimdDirective.html) functionality.
    }];

    let methods = [];
}

def VAST_OMPTargetSimdDirectiveInterface : VAST_ASTOpInterface< "OMPTargetSimdDirectiveInterface",
    [VAST_OMPLoopDirectiveInterface] > {

    let description = [{
        An interface to provide [`clang::OMPTargetSimdDirective`]
        (https://clang.llvm.org/doxygen/classclang_1_1OMPTargetSimdDirective.html) functionality.
    }];

    let methods = [];
}

def VAST_OMPTeamsDistributeDirectiveInterface : VAST_ASTOpInterface< "OMPTeamsDistributeDirectiveInterface",
    [VAST_OMPLoopDirectiveInterface] > {

    let description = [{
        An interface to provide [`clang::OMPTeamsDistributeDirective`]
        (https://clang.llvm.org/doxygen/classclang_1_1OMPTeamsDistributeDirective.html) functionality.
    }];

    let methods = [];
}

def VAST_OMPTeamsDistributeSimdDirectiveInterface : VAST_ASTOpInterface< "OMPTeamsDistributeSimdDirectiveInterface",
    [VAST_OMPLoopDirectiveInterface] > {

    let description = [{
        An interface to provide [`clang::OMPTeamsDistributeSimdDirective`]
        (https://clang.llvm.org/doxygen/classclang_1_1OMPTeamsDistributeSimdDirective.html) functionality.
    }];

    let methods = [];
}

def VAST_OMPTeamsDistributeParallelForSimdDirectiveInterface : VAST_ASTOpInterface< "OMPTeamsDistributeParallelForSimdDirectiveInterface",
    [VAST_OMPLoopDirectiveInterface] > {

    let description = [{
        An interface to provide [`clang::OMPTeamsDistributeParallelForSimdDirective`]
        (https://clang.llvm.org/doxygen/classclang_1_1OMPTeamsDistributeParallelForSimdDirective.html) functionality.
    }];

    let methods = [];
}

def VAST_OMPTeamsDistributeParallelForDirectiveInterface : VAST_ASTOpInterface< "OMPTeamsDistributeParallelForDirectiveInterface",
    [VAST_OMPLoopDirectiveInterface] > {

    let description = [{
        An interface to provide [`clang::OMPTeamsDistributeParallelForDirective`]
        (https://clang.llvm.org/doxygen/classclang_1_1OMPTeamsDistributeParallelForDirective.html) functionality.
    }];

    let methods = [];
}

def VAST_OMPTargetTeamsDirectiveInterface : VAST_ASTOpInterface< "OMPTargetTeamsDirectiveInterface",
    [VAST_OMPExecutableDirectiveInterface] > {

    let description = [{
        An interface to provide [`clang::OMPTargetTeamsDirective`]
        (https://clang.llvm.org/doxygen/classclang_1_1OMPTargetTeamsDirective.html) functionality.
    }];

    let methods = [];
}

def VAST_OMPTargetTeamsDistributeDirectiveInterface : VAST_ASTOpInterface< "OMPTargetTeamsDistributeDirectiveInterface",
    [VAST_OMPLoopDirectiveInterface] > {

    let description = [{
        An interface to provide [`clang::OMPTargetTeamsDistributeDirective`]
        (https://clang.llvm.org/doxygen/classclang_1_1OMPTargetTeamsDistributeDirective.html) functionality.
    }];

    let methods = [];
}

def VAST_OMPTargetTeamsDistributeParallelForDirectiveInterface : VAST_ASTOpInterface< "OMPTargetTeamsDistributeParallelForDirectiveInterface",
    [VAST_OMPLoopDirectiveInterface] > {

    let description = [{
        An interface to provide [`clang::OMPTargetTeamsDistributeParallelForDirective`]
        (https://clang.llvm.org/doxygen/classclang_1_1OMPTargetTeamsDistributeParallelForDirective.html) functionality.
    }];

    let methods = [];
}

def VAST_OMPTargetTeamsDistributeParallelForSimdDirectiveInterface : VAST_ASTOpInterface< "OMPTargetTeamsDistributeParallelForSimdDirectiveInterface",
    [VAST_OMPLoopDirectiveInterface] > {

    let description = [{
        An interface to provide [`clang::OMPTargetTeamsDistributeParallelForSimdDirective`]
        (https://clang.llvm.org/doxygen/classclang_1_1OMPTargetTeamsDistributeParallelForSimdDirective.html) functionality.
    }];

    let methods = [];
}

def VAST_OMPTargetTeamsDistributeSimdDirectiveInterface : VAST_ASTOpInterface< "OMPTargetTeamsDistributeSimdDirectiveInterface",
    [VAST_OMPLoopDirectiveInterface] > {

    let description = [{
        An interface to provide [`clang::OMPTargetTeamsDistributeSimdDirective`]
        (https://clang.llvm.org/doxygen/classclang_1_1OMPTargetTeamsDistributeSimdDirective.html) functionality.
    }];

    let methods = [];
}

def VAST_OMPInteropDirectiveInterface : VAST_ASTOpInterface< "OMPInteropDirectiveInterface",
    [VAST_OMPExecutableDirectiveInterface] > {

    let description = [{
        An interface to provide [`clang::OMPInteropDirective`]
        (https://clang.llvm.org/doxygen/classclang_1_1OMPInteropDirective.html) functionality.
    }];

    let methods = [];
}

def VAST_OMPDispatchDirectiveInterface : VAST_ASTOpInterface< "OMPDispatchDirectiveInterface",
    [VAST_OMPExecutableDirectiveInterface] > {

    let description = [{
        An interface to provide [`clang::OMPDispatchDirective`]
        (https://clang.llvm.org/doxygen/classclang_1_1OMPDispatchDirective.html) functionality.
    }];

    let methods = [];
}

def VAST_OMPMaskedDirectiveInterface : VAST_ASTOpInterface< "OMPMaskedDirectiveInterface",
    [VAST_OMPExecutableDirectiveInterface] > {

    let description = [{
        An interface to provide [`clang::OMPMaskedDirective`]
        (https://clang.llvm.org/doxygen/classclang_1_1OMPMaskedDirective.html) functionality.
    }];

    let methods = [];
}

def VAST_OMPParallelMaskedDirectiveInterface : VAST_ASTOpInterface< "OMPParallelMaskedDirectiveInterface",
    [VAST_OMPExecutableDirectiveInterface] > {

    let description = [{
        An interface to provide [`clang::OMPParallelMaskedDirective`]
        (https://clang.llvm.org/doxygen/classclang_1_1OMPParallelMaskedDirective.html) functionality.
    }];

    let methods = [];
}

def VAST_OMPGenericLoopDirectiveInterface : VAST_ASTOpInterface< "OMPGenericLoopDirectiveInterface",
    [VAST_OMPLoopDirectiveInterface] > {

    let description = [{
        An interface to provide [`clang::OMPGenericLoopDirective`]
        (https://clang.llvm.org/doxygen/classclang_1_1OMPGenericLoopDirective.html) functionality.
    }];

    let methods = [];
}

def VAST_OMPTeamsGenericLoopDirectiveInterface : VAST_ASTOpInterface< "OMPTeamsGenericLoopDirectiveInterface",
    [VAST_OMPLoopDirectiveInterface] > {

    let description = [{
        An interface to provide [`clang::OMPTeamsGenericLoopDirective`]
        (https://clang.llvm.org/doxygen/classclang_1_1OMPTeamsGenericLoopDirective.html) functionality.
    }];

    let methods = [];
}

def VAST_OMPTargetTeamsGenericLoopDirectiveInterface : VAST_ASTOpInterface< "OMPTargetTeamsGenericLoopDirectiveInterface",
    [VAST_OMPLoopDirectiveInterface] > {

    let description = [{
        An interface to provide [`clang::OMPTargetTeamsGenericLoopDirective`]
        (https://clang.llvm.org/doxygen/classclang_1_1OMPTargetTeamsGenericLoopDirective.html) functionality.
    }];

    let methods = [];
}

def VAST_OMPParallelGenericLoopDirectiveInterface : VAST_ASTOpInterface< "OMPParallelGenericLoopDirectiveInterface",
    [VAST_OMPLoopDirectiveInterface] > {

    let description = [{
        An interface to provide [`clang::OMPParallelGenericLoopDirective`]
        (https://clang.llvm.org/doxygen/classclang_1_1OMPParallelGenericLoopDirective.html) functionality.
    }];

    let methods = [];
}

def VAST_OMPTargetParallelGenericLoopDirectiveInterface : VAST_ASTOpInterface< "OMPTargetParallelGenericLoopDirectiveInterface",
    [VAST_OMPLoopDirectiveInterface] > {

    let description = [{
        An interface to provide [`clang::OMPTargetParallelGenericLoopDirective`]
        (https://clang.llvm.org/doxygen/classclang_1_1OMPTargetParallelGenericLoopDirective.html) functionality.
    }];

    let methods = [];
}

def VAST_OMPErrorDirectiveInterface : VAST_ASTOpInterface< "OMPErrorDirectiveInterface",
    [VAST_OMPExecutableDirectiveInterface] > {

    let description = [{
        An interface to provide [`clang::OMPErrorDirective`]
        (https://clang.llvm.org/doxygen/classclang_1_1OMPErrorDirective.html) functionality.
    }];

    let methods = [];
}

def VAST_OpenACCConstructStmtInterface : VAST_ASTOpInterface< "OpenACCConstructStmtInterface",
    [VAST_StmtInterface] > {

    let description = [{
        An interface to provide [`clang::OpenACCConstructStmt`]
        (https://clang.llvm.org/doxygen/classclang_1_1OpenACCConstructStmt.html) functionality.
    }];

    let methods = [];
}

def VAST_OpenACCAssociatedStmtConstructInterface : VAST_ASTOpInterface< "OpenACCAssociatedStmtConstructInterface",
    [VAST_OpenACCConstructStmtInterface] > {

    let description = [{
        An interface to provide [`clang::OpenACCAssociatedStmtConstruct`]
        (https://clang.llvm.org/doxygen/classclang_1_1OpenACCAssociatedStmtConstruct.html) functionality.
    }];

    let methods = [];
}

def VAST_OpenACCComputeConstructInterface : VAST_ASTOpInterface< "OpenACCComputeConstructInterface",
    [VAST_OpenACCAssociatedStmtConstructInterface] > {

    let description = [{
        An interface to provide [`clang::OpenACCComputeConstruct`]
        (https://clang.llvm.org/doxygen/classclang_1_1OpenACCComputeConstruct.html) functionality.
    }];

    let methods = [];
}

def VAST_OpenACCLoopConstructInterface : VAST_ASTOpInterface< "OpenACCLoopConstructInterface",
    [VAST_OpenACCAssociatedStmtConstructInterface] > {

    let description = [{
        An interface to provide [`clang::OpenACCLoopConstruct`]
        (https://clang.llvm.org/doxygen/classclang_1_1OpenACCLoopConstruct.html) functionality.
    }];

    let methods = [];
}

#endif // VAST_INTERFACES_AST_STMT_INTERFACE
