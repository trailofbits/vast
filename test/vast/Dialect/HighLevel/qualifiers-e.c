// RUN: %vast-cc --ccopts -xc --from-source %s | FileCheck %s
// RUN: %vast-cc --ccopts -xc --from-source %s > %t && %vast-opt %t | diff -B %t -

// CHECK: hl.var "ia" : !hl.lvalue<!hl.array<10, !hl.int>>
int ia[10];

// CHECK: hl.var "cia" : !hl.lvalue<!hl.array<10, !hl.int< const >>>
const int cia[10];

// CHECK: hl.var "via" : !hl.lvalue<!hl.array<10, !hl.int< volatile >>>
volatile int via[10];

// CHECK: hl.var "cvia" : !hl.lvalue<!hl.array<10, !hl.int< const, volatile >>>
const volatile int cvia[10];
