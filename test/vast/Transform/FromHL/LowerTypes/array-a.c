// RUN: vast-cc --ccopts -xc --from-source %s | vast-opt --vast-hl-lower-types | FileCheck %s

// CHECK: hl.var "ai" : !hl.lvalue<memref<10xi32>>
int ai[10];

// CHECK: hl.var "aci" : !hl.lvalue<memref<5xi32>>
const int aci[5];

// CHECK: hl.var "avi" : !hl.lvalue<memref<5xi32>>
volatile int avi[5];

// CHECK: hl.var "acvi" : !hl.lvalue<memref<5xi32>>
const volatile int acvi[5];

// CHECK: hl.var "acvui" : !hl.lvalue<memref<5xi32>>
const volatile unsigned int acvui[5];

// CHECK: hl.var "af" : !hl.lvalue<memref<10xf32>>
float af[10];

// CHECK: hl.var "a3d" : !hl.lvalue<memref<2x4x3xf32>>
float a3d[2][4][3];
