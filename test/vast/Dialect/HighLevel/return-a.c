// RUN: %vast-cc --ccopts -xc --from-source %s | FileCheck %s
// RUN: %vast-cc --ccopts -xc --from-source %s > %t && %vast-opt %t | diff -B %t -

// CHECK-LABEL: hl.func @a () -> !hl.int
// CHECK: [[V1:%[0-9]+]] = hl.const #hl.integer<7> : !hl.int
// CHECK: hl.return [[V1]] : !hl.int
int a() { return 7; }

// CHECK-LABEL: hl.func @b ()
// CHECK: hl.return
void b() { return; }
