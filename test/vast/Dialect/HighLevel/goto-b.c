// RUN: vast-cc --from-source %s | FileCheck %s
// RUN: vast-cc --from-source %s > %t && vast-opt %t | diff -B %t -
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
