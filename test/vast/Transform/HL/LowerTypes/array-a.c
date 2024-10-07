// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %vast-opt --vast-hl-lower-types | %file-check %s

// CHECK: hl.var @ai, <external> : !hl.lvalue<!hl.array<10, si32>>
int ai[10];

// CHECK: hl.var @aci, <external> : !hl.lvalue<!hl.array<5, si32>>
const int aci[5];

// CHECK: hl.var @avi, <external> : !hl.lvalue<!hl.array<5, si32>>
volatile int avi[5];

// CHECK: hl.var @acvi, <external> : !hl.lvalue<!hl.array<5, si32>>
const volatile int acvi[5];

// CHECK: hl.var @acvui, <external> : !hl.lvalue<!hl.array<5, ui32>>
const volatile unsigned int acvui[5];

// CHECK: hl.var @af, <external> : !hl.lvalue<!hl.array<10, f32>>
float af[10];

// CHECK: hl.var @a3d, <external> : !hl.lvalue<!hl.array<2, !hl.array<4, !hl.array<3, f32>>>>
float a3d[2][4][3];
