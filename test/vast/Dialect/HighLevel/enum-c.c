// RUN: vast-cc --ccopts -xc --from-source %s | FileCheck %s
// RUN: vast-cc --ccopts -xc --from-source %s > %t && vast-opt %t | diff -B %t -

// CHECK: hl.enum "Foo" : !hl.int< unsigned >
// CHECK:  hl.enum.const "A" = #hl.integer<0> : !hl.int
// CHECK:  hl.enum.const "B" = #hl.integer<1> : !hl.int
// CHECK:  hl.enum.const "C" = #hl.integer<10> : !hl.int init
// CHECK:   hl.const #hl.integer<10> : !hl.int
// CHECK:  hl.enum.const "D" = #hl.integer<11> : !hl.int
// CHECK:  hl.enum.const "E" = #hl.integer<1> : !hl.int init
// CHECK:   hl.const #hl.integer<1> : !hl.int
// CHECK:  hl.enum.const "F" = #hl.integer<2> : !hl.int
// CHECK:  hl.enum.const "G" = #hl.integer<12> : !hl.int init
// CHECK:   [[V1:%[0-9]+]] = hl.enumref "F" : !hl.int
// CHECK:   [[V2:%[0-9]+]] = hl.enumref "C" : !hl.int
// CHECK:   [[V3:%[0-9]+]] = hl.add [[V1]], [[V2]] : (!hl.int, !hl.int) -> !hl.int
enum Foo { A, B, C=10, D, E=1, F, G=F+C };
