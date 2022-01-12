// RUN: vast-cc --ccopts -xc --from-source %s | FileCheck %s
// RUN: vast-cc --ccopts -xc --from-source %s > %t && vast-opt --vast-hl-lower-types %t | diff -B %t -

// CHECK: hl.global @ai : memref<10xi32>
int ai[10];

// CHECK: hl.global @aci : memref<5xi32>
const int aci[5];

// CHECK: hl.global @avi : memref<5xi32>
volatile int avi[5];

// CHECK: hl.global @acvi : memref<5xi32>
const volatile int acvi[5];

// CHECK: hl.global @acvui : memref<5xi32>
const volatile unsigned int acvui[5];

// CHECK: hl.global @af : memref<10xf32>
float af[10];

// CHECK: hl.global @a3d : memref<2x4x3xf32>
float a3d[2][4][3];
