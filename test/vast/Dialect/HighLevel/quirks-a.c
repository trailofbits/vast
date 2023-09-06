// RUN: %vast-cc --ccopts -xc --from-source %s | FileCheck %s
// RUN: %vast-cc --ccopts -xc --from-source %s > %t && %vast-opt %t | diff -B %t -

// adapted from https://gist.github.com/fay59/5ccbe684e6e56a7df8815c3486568f01

// CHECK: hl.struct "foo"
struct foo {
    // CHECK: hl.struct "bar"
    struct bar {
        int x;
    } baz;
};

void frob() {
    // CHECK: hl.var "b" : !hl.lvalue<!hl.elaborated<!hl.record<"bar">>>
    struct bar b;
}
