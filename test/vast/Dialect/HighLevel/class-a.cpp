// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -
// REQUIRES: c++


// CHECK: hl.class "A" :
class A {};

// CHECK: hl.class "B" :
class B {};

// CHECK: hl.class "C" :
class C : public A, protected virtual B {
    // CHECK: hl.base !hl.elaborated<!hl.record<"A">> public
    // CHECK: hl.base !hl.elaborated<!hl.record<"B">> protected virtual

// CHECK: hl.access public
public:
    // CHECK: hl.field "x" : !hl.int
    int x;

    // CHECK: %0 = hl.var @C::y sc_static : !hl.lvalue<!hl.int>
    static int y;
};
