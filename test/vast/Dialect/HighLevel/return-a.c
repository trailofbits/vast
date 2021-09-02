// RUN: vast-cc --from-source %s | FileCheck %s

int a() { return 7; }
// CHECK-LABEL: func private @a() -> !hl.int
// CHECK: hl.constant( 7 : i32 ): !hl.int
// CHECK: return %0 : !hl.int
