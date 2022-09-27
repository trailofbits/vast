// RUN: vast-cc --from-source %s | FileCheck %s
// RUN: vast-cc --from-source %s > %t && vast-opt %t | diff -B %t -

// CHECK-LABEL: func @a() -> !hl.int
// CHECK: [[V1:%[0-9]+]] = hl.const #hl.integer<7> : !hl.int
// CHECK: hl.return [[V1]] : !hl.int
int a() { return 7; }

// CHECK-LABEL: func @b() -> !hl.void
// CHECK: hl.return
void b() { return; }
