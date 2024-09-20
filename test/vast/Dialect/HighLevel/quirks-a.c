// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

// adapted from https://gist.github.com/fay59/5ccbe684e6e56a7df8815c3486568f01

// CHECK: hl.struct @foo
struct foo {
    // CHECK: hl.struct @bar
    struct bar {
        int x;
    } baz;
};

void frob() {
    // CHECK: hl.var @b : !hl.lvalue<!hl.elaborated<!hl.record<@bar>>>
    struct bar b;
}
