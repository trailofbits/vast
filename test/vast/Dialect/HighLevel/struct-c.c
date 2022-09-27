// RUN: vast-cc --ccopts -xc --from-source %s | FileCheck %s
// RUN: vast-cc --ccopts -xc --from-source %s > %t && vast-opt %t | diff -B %t -

// CHECK: hl.struct "anonymous[512]" : {
// CHECK:  hl.field "data" : !hl.int
// CHECK: }
// CHECK: hl.var "named" : !hl.lvalue<!hl.elaborated<!hl.record<"anonymous[512]">>>
struct {
  int data;
} named;
