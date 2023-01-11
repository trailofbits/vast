// RUN: vast-cc --ccopts -xc --from-source %s | FileCheck %s
// RUN: vast-cc --ccopts -xc --from-source %s > %t && vast-opt %t | diff -B %t -

// adapted from https://gist.github.com/fay59/5ccbe684e6e56a7df8815c3486568f01

// CHECK: hl.func external @foo
int foo() {
    // CHECK: hl.var "a" : !hl.lvalue<!hl.ptr<!hl.int>>
    int* a;
    // CHECK: hl.var "b" : !hl.lvalue<!hl.ptr<!hl.int>>
    int* b;
    // CHECK: hl.scope
    {
        // CHECK: hl.var "foo" sc_static : !hl.lvalue<!hl.int>
        static int foo;
        a = &foo;
    }
    // CHECK: hl.scope
    {
        // CHECK: hl.var "foo" sc_static : !hl.lvalue<!hl.int>
        static int foo;
        b = &foo;
    }
    // this always returns false: two static variables with the same name
    // but declared in different scope refer to different storage.
    return a == b;
}
