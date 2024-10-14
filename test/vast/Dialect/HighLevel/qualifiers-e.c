// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

// CHECK: hl.var @ia, <external> : !hl.lvalue<!hl.array<10, !hl.int>>
int ia[10];

// CHECK: hl.var @cia, <external> constant : !hl.lvalue<!hl.array<10, !hl.int< const >>>
const int cia[10];

// CHECK: hl.var @via, <external> : !hl.lvalue<!hl.array<10, !hl.int< volatile >>>
volatile int via[10];

// CHECK: hl.var @cvia, <external> constant : !hl.lvalue<!hl.array<10, !hl.int< const, volatile >>>
const volatile int cvia[10];
