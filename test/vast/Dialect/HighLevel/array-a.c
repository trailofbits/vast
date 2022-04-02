// RUN: vast-cc --ccopts -xc --from-source %s | FileCheck %s
// RUN: vast-cc --ccopts -xc --from-source %s > %t && vast-opt %t | diff -B %t -

// CHECK: hl.var "ai" : !hl.lvalue<!hl.const.array<10, !hl.int>>
int ai[10];

// CHECK: hl.var "aci" : !hl.lvalue<!hl.const.array<5, !hl.int<const>, const>>
const int aci[5];

// CHECK: hl.var "avi" : !hl.lvalue<!hl.const.array<5, !hl.int<volatile>, volatile>>
volatile int avi[5];

// CHECK: hl.var "acvi" : !hl.lvalue<!hl.const.array<5, !hl.int<const volatile>, const volatile>>
const volatile int acvi[5];

// CHECK: hl.var "acvui" : !hl.lvalue<!hl.const.array<5, !hl.int<unsigned const volatile>, const volatile>>
const volatile unsigned int acvui[5];

// CHECK: hl.var "af" : !hl.lvalue<!hl.const.array<10, !hl.float>>
float af[10];

// CHECK: hl.var "a3d" : !hl.lvalue<!hl.const.array<2, !hl.const.array<4, !hl.const.array<3, !hl.float>>>>
float a3d[2][4][3];
