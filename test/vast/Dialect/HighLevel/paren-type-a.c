// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

void fun() {
    int (a) = 0;
    // CHECK: hl.add %{{[0-9]+}}, %{{[0-9]+}} : (!hl.paren<!hl.int>, !hl.int)
    a + 0;
    float (b) = 0;
    // CHECK: hl.fadd %{{[0-9]+}}, %{{[0-9]+}} : (!hl.paren<!hl.float>, !hl.float)
    b + 0.0f;
}

enum FOO { BAR };

void fun2 () {
    int a[1]    = { 0 };
    enum FOO (b) = BAR;
    // CHECK: hl.subscript %{{[0-9]+}} at [%{{[0-9]+}} : !hl.paren<!hl.elaborated<!hl.enum<@FOO>>>]
    a[b];
}
