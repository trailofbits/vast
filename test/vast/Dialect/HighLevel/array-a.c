// RUN: vast-cc --ccopts -xc --from-source %s | FileCheck %s
// RUN: vast-cc --ccopts -xc --from-source %s > %t && vast-opt %t | diff -B %t -

// CHECK: hl.global @ai : !hl.const.array<10, !hl.int>
int ai[10];

// CHECK: hl.global @aci : !hl.const.array<5, !hl.int<const>, const>
const int aci[5];

// CHECK: hl.global @avi : !hl.const.array<5, !hl.int<volatile>, volatile>
volatile int avi[5];

// CHECK: hl.global @acvi : !hl.const.array<5, !hl.int<const volatile>, const volatile>
const volatile int acvi[5];

// CHECK: hl.global @acvui : !hl.const.array<5, !hl.int<unsigned const volatile>, const volatile>
const volatile unsigned int acvui[5];

// CHECK: hl.global @af : !hl.const.array<10, !hl.float>
float af[10];

// CHECK: hl.global @a3d : !hl.const.array<2, !hl.const.array<4, !hl.const.array<3, !hl.float>>>
float a3d[2][4][3];
