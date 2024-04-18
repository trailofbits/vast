// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

// CHECK-LABEL: hl.func @a {{.*}} () -> !hl.int
// CHECK: [[V1:%[0-9]+]] = hl.const #core.integer<7> : !hl.int
// CHECK: hl.return [[V1]] : !hl.int
int a() { return 7; }

// CHECK-LABEL: hl.func @b {{.*}} ()
// CHECK: hl.return
void b() { return; }
