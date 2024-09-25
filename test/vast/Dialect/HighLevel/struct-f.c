// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

// CHECK: hl.struct @X
struct X {};

// CHECK: hl.struct @Y
// CHECK: hl.typedef @X : !hl.elaborated<!hl.record<@Y>>
typedef struct Y {} X;

// CHECK: hl.var @x, <common> : !hl.lvalue<!hl.elaborated<!hl.record<@X>>>
// TODO: this is elaborated "X"
struct X x;

// CHECK: hl.typedef @Y : !hl.elaborated<!hl.typedef<@X>>
typedef X Y;
