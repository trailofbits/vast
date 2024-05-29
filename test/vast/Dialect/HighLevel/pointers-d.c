// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

void f(int);
// CHECK: hl.var @pf1 : !hl.lvalue<!hl.ptr<!hl.paren<!core.fn<(!hl.lvalue<!hl.int>) -> (!hl.void)>>>>
// CHECK:   [[R:%[0-9]+]] = hl.funcref @f : !core.fn<(!hl.lvalue<!hl.int>) -> (!hl.void)>
// CHECK:   hl.addressof [[R]] : !core.fn<(!hl.lvalue<!hl.int>) -> (!hl.void)> -> !hl.ptr<!core.fn<(!hl.lvalue<!hl.int>) -> (!hl.void)>>
void (*pf1)(int) = &f;
// CHECK: hl.var @pf2 : !hl.lvalue<!hl.ptr<!hl.paren<!core.fn<(!hl.lvalue<!hl.int>) -> (!hl.void)>>>>
// CHECK:   [[R:%[0-9]+]] = hl.funcref @f : !core.fn<(!hl.lvalue<!hl.int>) -> (!hl.void)>
// CHECK:   hl.implicit_cast [[R]] FunctionToPointerDecay : !core.fn<(!hl.lvalue<!hl.int>) -> (!hl.void)> -> !hl.ptr<!core.fn<(!hl.lvalue<!hl.int>) -> (!hl.void)>>
void (*pf2)(int) = f; // same as &f
