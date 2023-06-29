// RUN: vast-cc --ccopts -xc --from-source %s | FileCheck %s
// RUN: vast-cc --ccopts -xc --from-source %s > %t && vast-opt %t | diff -B %t -

// CHECK: hl.var "i" : !hl.lvalue<!hl.int>
// CHECK:   hl.const #hl.integer<0> : !hl.int
int i = 0;

// CHECK: hl.var "j" : !hl.lvalue<!hl.typeofexpr<!hl.int>>
// CHECK:   hl.const #hl.integer<0> : !hl.int
typeof(i) j = 0;