// RUN: vast-cc --ccopts -xc --from-source %s | FileCheck %s
// RUN: vast-cc --ccopts -xc --from-source %s > %t && vast-opt %t | diff -B %t -

// CHECK: hl.enum.decl @enum.Foo : !hl.int<unsigned>
// CHECK:  hl.enum.const @A = 0 : si32
// CHECK:  hl.enum.const @B = 1 : si32
// CHECK:  hl.enum.const @C = 10 : si32 init
// CHECK:   hl.constant.int 10 : !hl.int
// CHECK:  hl.enum.const @D = 11 : si32
// CHECK:  hl.enum.const @E = 1 : si32 init
// CHECK:   hl.constant.int 1 : !hl.int
// CHECK:  hl.enum.const @F = 2 : si32
// CHECK:  hl.enum.const @G = 12 : si32 init
// CHECK:   [[V1:%[0-9]+]] = hl.declref @F : !hl.int
// CHECK:   [[V2:%[0-9]+]] = hl.declref @C : !hl.int
// CHECK:   [[V3:%[0-9]+]] = hl.add %0, %1 : !hl.int
enum Foo { A, B, C=10, D, E=1, F, G=F+C };
