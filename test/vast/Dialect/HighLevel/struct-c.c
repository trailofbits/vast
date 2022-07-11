// RUN: vast-cc --ccopts -xc --from-source %s | FileCheck %s
// RUN: vast-cc --ccopts -xc --from-source %s > %t && vast-opt %t | diff -B %t -

// CHECK: hl.struct "struct anonymous.0" : {
// CHECK:  hl.field "data" : !hl.int
// CHECK: }
// CHECK: hl.var "named" : !hl.lvalue<!hl.named_type<"struct anonymous.0">>
struct {
  int data;
} named;
