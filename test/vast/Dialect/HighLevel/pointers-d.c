// RUN: %vast-cc --ccopts -xc --from-source %s | FileCheck %s
// RUN: %vast-cc --ccopts -xc --from-source %s > %t && %vast-opt %t | diff -B %t -

void f(int);
// CHECK: hl.var "pf1" : !hl.lvalue<!hl.ptr<!hl.paren<(!hl.lvalue<!hl.int>) -> !hl.void>>>
// CHECK:   [[R:%[0-9]+]] = hl.funcref @f : !hl.lvalue<(!hl.lvalue<!hl.int>) -> !hl.void>
// CHECK:   hl.addressof [[R]] : !hl.lvalue<(!hl.lvalue<!hl.int>) -> !hl.void> -> !hl.ptr<(!hl.lvalue<!hl.int>) -> !hl.void>
void (*pf1)(int) = &f;
// CHECK: hl.var "pf2" : !hl.lvalue<!hl.ptr<!hl.paren<(!hl.lvalue<!hl.int>) -> !hl.void>>>
// CHECK:   [[R:%[0-9]+]] = hl.funcref @f : !hl.lvalue<(!hl.lvalue<!hl.int>) -> !hl.void>
// CHECK:   hl.implicit_cast [[R]] FunctionToPointerDecay : !hl.lvalue<(!hl.lvalue<!hl.int>) -> !hl.void> -> !hl.lvalue<!hl.ptr<(!hl.lvalue<!hl.int>) -> !hl.void>>
void (*pf2)(int) = f; // same as &f
