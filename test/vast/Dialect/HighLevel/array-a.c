// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

// CHECK: hl.var @ai, <external> : !hl.lvalue<!hl.array<10, !hl.int>>
int ai[10];

// CHECK: hl.var @aci, <external> constant : !hl.lvalue<!hl.array<5, !hl.int< const >>>
const int aci[5];

// CHECK: hl.var @avi, <external> : !hl.lvalue<!hl.array<5, !hl.int< volatile >>>
volatile int avi[5];

// CHECK: hl.var @acvi, <external> constant : !hl.lvalue<!hl.array<5, !hl.int< const, volatile >>>
const volatile int acvi[5];

// CHECK: hl.var @acvui, <external> constant : !hl.lvalue<!hl.array<5, !hl.int< unsigned, const, volatile >>>
const volatile unsigned int acvui[5];

// CHECK: hl.var @af, <external> : !hl.lvalue<!hl.array<10, !hl.float>>
float af[10];

// CHECK: hl.var @a3d, <external> : !hl.lvalue<!hl.array<2, !hl.array<4, !hl.array<3, !hl.float>>>>
float a3d[2][4][3];

// CHECK: hl.var @ae, <external> : !hl.lvalue<!hl.array<404, !hl.int>>
int ae[4 + 4*100];
