// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

int f();

int main() {
    // CHECK: [[FP:%[0-9]+]] = hl.var @p : !hl.lvalue<!hl.ptr<!hl.paren<!core.fn<() -> (!hl.int)>>>>
    // CHECK:   hl.funcref @f : !core.fn<() -> (!hl.int)>
    // CHECK:   FunctionToPointerDecay : !core.fn<() -> (!hl.int)> -> !hl.ptr<!core.fn<() -> (!hl.int)>>
    int (*p)() = f;    // pointer p is pointing to f

    // CHECK: [[E:%[0-9]+]] = hl.expr : !hl.paren<!core.fn<() -> (!hl.int)>>
    // CHECK: [[P:%[0-9]+]] = hl.implicit_cast [[E]] FunctionToPointerDecay : !hl.paren<!core.fn<() -> (!hl.int)>> -> !hl.ptr<!hl.paren<!core.fn<() -> (!hl.int)>>>
    // CHECK: hl.indirect_call [[P]] : !hl.ptr<!hl.paren<!core.fn<() -> (!hl.int)>>>() : () -> !hl.int
    (*p)(); // function f invoked through the function designator

    // CHECK: [[R:%[0-9]+]] = hl.ref [[FP]]
    // CHECK: [[F:%[0-9]+]] = hl.implicit_cast [[R]] LValueToRValue
    // CHECK:  hl.indirect_call [[F]] : !hl.ptr<!hl.paren<!core.fn<() -> (!hl.int)>>>() : () -> !hl.int
    p();    // function f invoked directly through the pointer
}
