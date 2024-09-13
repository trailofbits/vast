// RUN: %vast-front -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-front -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

void foo(int test) {

    // CHECK: hl.var
    void *ptr;

    if (test)
    // CHECK: hl.labeladdr
        ptr = &&foo;
    else
    // CHECK: hl.labeladdr
        ptr = &&bar;

    // CHECK: hl.indirect_goto : {
    // CHECK: hl.ref @ptr
    // CHECK: }
    goto *ptr;

    foo: /* ... */;

    bar: /* ... */;
}
