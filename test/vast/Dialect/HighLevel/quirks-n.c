// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

// adapted from https://gist.github.com/fay59/5ccbe684e6e56a7df8815c3486568f01

// CHECK: hl.typedef @s16 : !hl.short
short typedef signed s16;
// CHECK: hl.typedef @u32 : !hl.int< unsigned >
unsigned int typedef u32;
// CHECK: hl.struct @foo
// CHECK:   hl.field @bar : !hl.int
// CHECK: hl.typedef @baz : !hl.elaborated<!hl.record<@foo>,  const >
struct foo { int bar; } const typedef baz;

// CHECK: hl.var @a, <external> : !hl.lvalue<!hl.elaborated<!hl.typedef<@s16>>>
s16 a;
// CHECK: hl.var @b, <external> : !hl.lvalue<!hl.elaborated<!hl.typedef<@u32>>>
u32 b;
// CHECK: hl.var @c, <external> constant : !hl.lvalue<!hl.elaborated<!hl.typedef<@baz>>>
baz c;
