// RUN: vast-cc --ccopts -xc --from-source %s | FileCheck %s
// RUN: vast-cc --ccopts -xc --from-source %s > %t && vast-opt %t | diff -B %t -

int f();

int main() {
    // CHECK: [[FP:%[0-9]+]] = hl.var "p" : !hl.lvalue<!hl.ptr<!hl.paren<() -> !hl.int>>>
    // CHECK:   hl.funcref @f : !hl.lvalue<() -> !hl.int>
    // CHECK:   FunctionToPointerDecay : !hl.lvalue<() -> !hl.int> -> !hl.lvalue<!hl.ptr<() -> !hl.int>>
    int (*p)() = f;    // pointer p is pointing to f

    // CHECK: [[E:%[0-9]+]] = hl.expr : !hl.lvalue<!hl.paren<() -> !hl.int>>
    // CHECK: [[P:%[0-9]+]] = hl.implicit_cast [[E]] FunctionToPointerDecay : !hl.lvalue<!hl.paren<() -> !hl.int>> -> !hl.lvalue<!hl.ptr<!hl.paren<() -> !hl.int>>>
    // CHECK: hl.indirect_call [[P]] : !hl.lvalue<!hl.ptr<!hl.paren<() -> !hl.int>>>() : () -> !hl.int
    (*p)(); // function f invoked through the function designator

    // CHECK: [[R:%[0-9]+]] = hl.ref [[FP]] : !hl.lvalue<!hl.ptr<!hl.paren<() -> !hl.int>>>
    // CHECK: [[F:%[0-9]+]] = hl.implicit_cast [[R]] LValueToRValue : !hl.lvalue<!hl.ptr<!hl.paren<() -> !hl.int>>> -> !hl.ptr<!hl.paren<() -> !hl.int>>
    // CHECK:  hl.indirect_call [[F]] : !hl.ptr<!hl.paren<() -> !hl.int>>() : () -> !hl.int
    p();    // function f invoked directly through the pointer
}
