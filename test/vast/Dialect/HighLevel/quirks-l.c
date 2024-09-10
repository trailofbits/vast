// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

// adapted from https://gist.github.com/fay59/5ccbe684e6e56a7df8815c3486568f01

struct foo {
    char a;
    long b: 16;
    char c;
};

// `struct foo` has the alignment of its most-aligned member:
// `long b` has an alignment of 8...
// CHECK: hl.alignof.type !hl.elaborated<!hl.record<"foo">>
int alignof_foo = _Alignof(struct foo);

// ...but `long b: 16` is a bitfield, and is aligned on a char
// boundary.
// CHECK: hl.offsetof.expr type : !hl.elaborated<!hl.record<"foo">>, member : [#hl.offset_of_node<identifier : "c">]
int offsetof_c = __builtin_offsetof(struct foo, c);
