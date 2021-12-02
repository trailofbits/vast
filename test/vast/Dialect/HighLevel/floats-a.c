// RUN: vast-cc --from-source %s | FileCheck %s
// RUN: vast-cc --from-source %s > %t && vast-opt %t | diff -B %t -

// CHECK: !hl.half
__fp16 fp16;

// CHECK: !hl.float
float f;

// CHECK: !hl.double
double d;

// CHECK: !hl.longdouble
long double ld;
