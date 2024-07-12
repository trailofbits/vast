// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

void fun() {
    int (a) = 0;
    // CHECK: hl.add %{{[0-9]+}}, %{{[0-9]+}} : (!hl.int, !hl.int)
    a + 0;
}

enum FOO { BAR };

void fun2 () {
    int a[1]    = { 0 };
    enum FOO (b) = BAR;
    // CHECK: hl.subscript %{{[0-9]+}} at [%{{[0-9]+}} : !hl.elaborated<!hl.enum<"FOO">>]
    a[b];
}
