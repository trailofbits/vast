//===- StmtVisitor.h - Visitor for Stmt subclasses --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines the StmtVisitor and ConstStmtVisitor interfaces.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_VAST_AST_STMTVISITOR_H
#define LLVM_VAST_AST_STMTVISITOR_H

#include "vast/Interfaces/AST/ExprInterface.hpp"

#include "clang/AST/OperationKinds.h"
#include "clang/AST/Stmt.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include <utility>

namespace vast::ast {
/// StmtVisitorBase - This class implements a simple visitor for Stmt
/// subclasses. Since Expr derives from Stmt, this also includes support for
/// visiting Exprs.
template<template <typename> class Ptr, typename ImplClass, typename RetTy=void,
         class... ParamTys>
class StmtVisitorBase {
public:
// Zmazat PTR alebo
#define PTR(CLASS) ::vast::ast::CLASS ## Interface
// #define PTR(CLASS) typename Ptr<CLASS>::type
#define DISPATCH(NAME, CLASS) \
  return static_cast< ImplClass* >(this)->Visit ## NAME( \
    cast< ::vast::ast:: CLASS ## Interface >(S.getOperation()), std::forward< ParamTys >(P)...)

    // static_cast<PTR(CLASS)>(S), std::forward<ParamTys>(P)...)
    // cast< vast::ast ## CLASS ## Interface >(S.getOperation()), ...

// mlir::Operation * || [ ast::StmtInterface ]
  RetTy Visit(PTR(Stmt) S, ParamTys... P) {
    // If we have a binary expr, dispatch to the subcode of the binop.  A smart
    // optimizer (e.g. LLVM) will fold this comparison into the switch stmt
    // below.
// auto BinOp = dyn_cast< ast::BinaryOperatorInterface >(S.getOperation())
    if (auto BinOp = dyn_cast< ast::BinaryOperatorInterface >(S.getOperation())) {
      switch (BinOp.getOpcode()) {
      case clang::BO_PtrMemD:   DISPATCH(BinPtrMemD,   BinaryOperator);
      case clang::BO_PtrMemI:   DISPATCH(BinPtrMemI,   BinaryOperator);
      case clang::BO_Mul:       DISPATCH(BinMul,       BinaryOperator);
      case clang::BO_Div:       DISPATCH(BinDiv,       BinaryOperator);
      case clang::BO_Rem:       DISPATCH(BinRem,       BinaryOperator);
      case clang::BO_Add:       DISPATCH(BinAdd,       BinaryOperator);
      case clang::BO_Sub:       DISPATCH(BinSub,       BinaryOperator);
      case clang::BO_Shl:       DISPATCH(BinShl,       BinaryOperator);
      case clang::BO_Shr:       DISPATCH(BinShr,       BinaryOperator);

      case clang::BO_LT:        DISPATCH(BinLT,        BinaryOperator);
      case clang::BO_GT:        DISPATCH(BinGT,        BinaryOperator);
      case clang::BO_LE:        DISPATCH(BinLE,        BinaryOperator);
      case clang::BO_GE:        DISPATCH(BinGE,        BinaryOperator);
      case clang::BO_EQ:        DISPATCH(BinEQ,        BinaryOperator);
      case clang::BO_NE:        DISPATCH(BinNE,        BinaryOperator);
      case clang::BO_Cmp:       DISPATCH(BinCmp,       BinaryOperator);

      case clang::BO_And:       DISPATCH(BinAnd,       BinaryOperator);
      case clang::BO_Xor:       DISPATCH(BinXor,       BinaryOperator);
      case clang::BO_Or :       DISPATCH(BinOr,        BinaryOperator);
      case clang::BO_LAnd:      DISPATCH(BinLAnd,      BinaryOperator);
      case clang::BO_LOr :      DISPATCH(BinLOr,       BinaryOperator);
      case clang::BO_Assign:    DISPATCH(BinAssign,    BinaryOperator);
      case clang::BO_MulAssign: DISPATCH(BinMulAssign, CompoundAssignOperator);
      case clang::BO_DivAssign: DISPATCH(BinDivAssign, CompoundAssignOperator);
      case clang::BO_RemAssign: DISPATCH(BinRemAssign, CompoundAssignOperator);
      case clang::BO_AddAssign: DISPATCH(BinAddAssign, CompoundAssignOperator);
      case clang::BO_SubAssign: DISPATCH(BinSubAssign, CompoundAssignOperator);
      case clang::BO_ShlAssign: DISPATCH(BinShlAssign, CompoundAssignOperator);
      case clang::BO_ShrAssign: DISPATCH(BinShrAssign, CompoundAssignOperator);
      case clang::BO_AndAssign: DISPATCH(BinAndAssign, CompoundAssignOperator);
      case clang::BO_OrAssign:  DISPATCH(BinOrAssign,  CompoundAssignOperator);
      case clang::BO_XorAssign: DISPATCH(BinXorAssign, CompoundAssignOperator);
      case clang::BO_Comma:     DISPATCH(BinComma,     BinaryOperator);
      }
    } else if (auto UnOp = dyn_cast< ast::UnaryOperatorInterface >(S.getOperation())) {
      switch (UnOp.getOpcode()) {
      case clang::UO_PostInc:   DISPATCH(UnaryPostInc,   UnaryOperator);
      case clang::UO_PostDec:   DISPATCH(UnaryPostDec,   UnaryOperator);
      case clang::UO_PreInc:    DISPATCH(UnaryPreInc,    UnaryOperator);
      case clang::UO_PreDec:    DISPATCH(UnaryPreDec,    UnaryOperator);
      case clang::UO_AddrOf:    DISPATCH(UnaryAddrOf,    UnaryOperator);
      case clang::UO_Deref:     DISPATCH(UnaryDeref,     UnaryOperator);
      case clang::UO_Plus:      DISPATCH(UnaryPlus,      UnaryOperator);
      case clang::UO_Minus:     DISPATCH(UnaryMinus,     UnaryOperator);
      case clang::UO_Not:       DISPATCH(UnaryNot,       UnaryOperator);
      case clang::UO_LNot:      DISPATCH(UnaryLNot,      UnaryOperator);
      case clang::UO_Real:      DISPATCH(UnaryReal,      UnaryOperator);
      case clang::UO_Imag:      DISPATCH(UnaryImag,      UnaryOperator);
      case clang::UO_Extension: DISPATCH(UnaryExtension, UnaryOperator);
      case clang::UO_Coawait:   DISPATCH(UnaryCoawait,   UnaryOperator);
      }
    }

    // Top switch stmt: dispatch to VisitFooStmt for each FooStmt.
    switch (S.getStmtClass()) {
    default: llvm_unreachable("Unknown stmt kind!");
#define ABSTRACT_STMT(STMT)
#define STMT(CLASS, PARENT)                              \
    case clang::Stmt::CLASS ## Class: DISPATCH(CLASS, CLASS);
#include "clang/AST/StmtNodes.inc"
    }
  }

  // If the implementation chooses not to implement a certain visit method, fall
  // back on VisitExpr or whatever else is the superclass.
  // CLASS je Interface... upravit vhodne nazov
#define STMT(CLASS, PARENT)                                   \
  RetTy Visit ## CLASS(PTR(CLASS) S, ParamTys... P) { DISPATCH(PARENT, PARENT); }
#include "clang/AST/StmtNodes.inc"

  // If the implementation doesn't implement binary operator methods, fall back
  // on VisitBinaryOperator.
#define BINOP_FALLBACK(NAME) \
  RetTy VisitBin ## NAME(PTR(BinaryOperator) S, ParamTys... P) { \
    DISPATCH(BinaryOperator, BinaryOperator); \
  }
  BINOP_FALLBACK(PtrMemD)                    BINOP_FALLBACK(PtrMemI)
  BINOP_FALLBACK(Mul)   BINOP_FALLBACK(Div)  BINOP_FALLBACK(Rem)
  BINOP_FALLBACK(Add)   BINOP_FALLBACK(Sub)  BINOP_FALLBACK(Shl)
  BINOP_FALLBACK(Shr)

  BINOP_FALLBACK(LT)    BINOP_FALLBACK(GT)   BINOP_FALLBACK(LE)
  BINOP_FALLBACK(GE)    BINOP_FALLBACK(EQ)   BINOP_FALLBACK(NE)
  BINOP_FALLBACK(Cmp)

  BINOP_FALLBACK(And)   BINOP_FALLBACK(Xor)  BINOP_FALLBACK(Or)
  BINOP_FALLBACK(LAnd)  BINOP_FALLBACK(LOr)

  BINOP_FALLBACK(Assign)
  BINOP_FALLBACK(Comma)
#undef BINOP_FALLBACK

  // If the implementation doesn't implement compound assignment operator
  // methods, fall back on VisitCompoundAssignOperator.
#define CAO_FALLBACK(NAME) \
  RetTy VisitBin ## NAME(PTR(CompoundAssignOperator) S, ParamTys... P) { \
    DISPATCH(CompoundAssignOperator, CompoundAssignOperator); \
  }
  CAO_FALLBACK(MulAssign) CAO_FALLBACK(DivAssign) CAO_FALLBACK(RemAssign)
  CAO_FALLBACK(AddAssign) CAO_FALLBACK(SubAssign) CAO_FALLBACK(ShlAssign)
  CAO_FALLBACK(ShrAssign) CAO_FALLBACK(AndAssign) CAO_FALLBACK(OrAssign)
  CAO_FALLBACK(XorAssign)
#undef CAO_FALLBACK

  // If the implementation doesn't implement unary operator methods, fall back
  // on VisitUnaryOperator.
#define UNARYOP_FALLBACK(NAME) \
  RetTy VisitUnary ## NAME(PTR(UnaryOperator) S, ParamTys... P) { \
    DISPATCH(UnaryOperator, UnaryOperator);    \
  }
  UNARYOP_FALLBACK(PostInc)   UNARYOP_FALLBACK(PostDec)
  UNARYOP_FALLBACK(PreInc)    UNARYOP_FALLBACK(PreDec)
  UNARYOP_FALLBACK(AddrOf)    UNARYOP_FALLBACK(Deref)

  UNARYOP_FALLBACK(Plus)      UNARYOP_FALLBACK(Minus)
  UNARYOP_FALLBACK(Not)       UNARYOP_FALLBACK(LNot)
  UNARYOP_FALLBACK(Real)      UNARYOP_FALLBACK(Imag)
  UNARYOP_FALLBACK(Extension) UNARYOP_FALLBACK(Coawait)
#undef UNARYOP_FALLBACK

  // Base case, ignore it. :)
  RetTy VisitStmt(PTR(Stmt) Node, ParamTys... P) { return RetTy(); }

#undef PTR
#undef DISPATCH
};

/// StmtVisitor - This class implements a simple visitor for Stmt subclasses.
/// Since Expr derives from Stmt, this also includes support for visiting Exprs.
///
/// This class does not preserve constness of Stmt pointers (see also
/// ConstStmtVisitor).
template <typename ImplClass, typename RetTy = void, typename... ParamTys>
class StmtVisitor
    : public StmtVisitorBase<std::add_pointer, ImplClass, RetTy, ParamTys...> {
public:
    using base = StmtVisitorBase<std::add_pointer, ImplClass, RetTy, ParamTys...>;
};

/// ConstStmtVisitor - This class implements a simple visitor for Stmt
/// subclasses. Since Expr derives from Stmt, this also includes support for
/// visiting Exprs.
///
/// This class preserves constness of Stmt pointers (see also StmtVisitor).
template <typename ImplClass, typename RetTy = void, typename... ParamTys>
class ConstStmtVisitor : public StmtVisitorBase<llvm::make_const_ptr, ImplClass,
                                                RetTy, ParamTys...> {};

} // namespace vast::ast

#endif // LLVM_VAST_AST_STMTVISITOR_H
