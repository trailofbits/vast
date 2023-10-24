// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -
// REQUIRES: indirect-goto

void foo(int test) {
    void *ptr;

    if (test)
        ptr = &&foo;
    else
        ptr = &&bar;

    goto *ptr;

    foo: /* ... */;

    bar: /* ... */;
}
