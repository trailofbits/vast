// RUN: vast-cc --from-source %s | FileCheck %s
// RUN: vast-cc --from-source %s > %t && vast-opt %t | diff -B %t -

// CHECK-LABEL: func private @a() -> !hl.int
// CHECK: [[V1:%[0-9]+]] = hl.constant 7 : !hl.int
// CHECK: hl.return [[V1]] : !hl.int
int a() { return 7; }

// CHECK-LABEL: func private @b() -> !hl.void
// CHECK: hl.return
void b() { return; }
