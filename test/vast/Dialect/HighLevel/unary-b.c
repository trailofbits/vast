// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

int main() {
    // CHECK: [[A:%[0-9]+]] = hl.var @a : !hl.lvalue<!hl.int>
    int a = 1;
    // CHECK: [[B:%[0-9]+]] = hl.var @b : !hl.lvalue<!hl.int>
    // CHECK:   [[AR:%[0-9]+]] = hl.ref [[A]]
    // CHECK:   hl.pre.dec [[AR]] : !hl.lvalue<!hl.int> -> !hl.int
    int b = --a;
}
