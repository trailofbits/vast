// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

// CHECK: hl.enum @Foo : !hl.int< unsigned >
// CHECK:  hl.enum.const @A = #core.integer<0> : !hl.int
// CHECK:  hl.enum.const @B = #core.integer<1> : !hl.int
// CHECK:  hl.enum.const @C = #core.integer<10> : !hl.int init
// CHECK:   hl.const #core.integer<10> : !hl.int
// CHECK:  hl.enum.const @D = #core.integer<11> : !hl.int
// CHECK:  hl.enum.const @E = #core.integer<1> : !hl.int init
// CHECK:   hl.const #core.integer<1> : !hl.int
// CHECK:  hl.enum.const @F = #core.integer<2> : !hl.int
// CHECK:  hl.enum.const @G = #core.integer<12> : !hl.int init
// CHECK:   [[V1:%[0-9]+]] = hl.enumref @F : !hl.int
// CHECK:   [[V2:%[0-9]+]] = hl.enumref @C : !hl.int
// CHECK:   [[V3:%[0-9]+]] = hl.add [[V1]], [[V2]] : (!hl.int, !hl.int) -> !hl.int
enum Foo { A, B, C=10, D, E=1, F, G=F+C };
