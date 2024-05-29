// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

// adapted from https://gist.github.com/fay59/5ccbe684e6e56a7df8815c3486568f01

// CHECK: hl.func @_Z3foov
int foo() {
    // CHECK: hl.var @a : !hl.lvalue<!hl.ptr<!hl.int>>
    int* a;
    // CHECK: hl.var @b : !hl.lvalue<!hl.ptr<!hl.int>>
    int* b;
    // CHECK: core.scope
    {
        // CHECK: hl.var @foo sc_static : !hl.lvalue<!hl.int>
        static int foo;
        a = &foo;
    }
    // CHECK: core.scope
    {
        // CHECK: hl.var @foo sc_static : !hl.lvalue<!hl.int>
        static int foo;
        b = &foo;
    }
    // this always returns false: two static variables with the same name
    // but declared in different scope refer to different storage.
    return a == b;
}
