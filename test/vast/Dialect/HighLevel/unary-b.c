// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

int main() {
    // CHECK: hl.var @a : !hl.lvalue<!hl.int>
    int a = 1;
    // CHECK: hl.var @b : !hl.lvalue<!hl.int>
    // CHECK:   [[AR:%[0-9]+]] = hl.ref @a
    // CHECK:   hl.pre.dec [[AR]] : !hl.lvalue<!hl.int> -> !hl.int
    int b = --a;
}
