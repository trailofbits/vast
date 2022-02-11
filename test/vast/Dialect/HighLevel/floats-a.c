// RUN: vast-cc --from-source %s | FileCheck %s
// RUN: vast-cc --from-source %s > %t && vast-opt %t | diff -B %t -

// CHECK: hl.var @fp16 : !hl.half
__fp16 fp16 = 0.5;

// CHECK: hl.var @f : !hl.float
// CHECK: hl.constant.float 5.000000e-01 : !hl.float
float f = 0.5f;

// CHECK: hl.var @d : !hl.double
// CHECK: hl.constant.float 5.000000e-01 : !hl.double
double d = 0.5;

// CHECK: hl.var @ld : !hl.longdouble
// CHECK: hl.constant.float 5.000000e-01 : !hl.longdouble
long double ld = 0.5L;
