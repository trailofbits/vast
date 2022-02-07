// RUN: vast-cc --ccopts -xc --from-source %s | FileCheck %s
// RUN: vast-cc --ccopts -xc --from-source %s > %t && vast-opt %t | diff -B %t -

_Bool oposite_signs(int x, int y) {
    // CHECK: hl.expr : !hl.int
    // CHECK:  hl.expr : !hl.int
    // CHECK:   hl.declref @x : !hl.int
    // CHECK:   hl.declref @y : !hl.int
    // CHECK:   hl.bin.xor
    // CHECK:  hl.constant.int 0 : !hl.int
    // CHECK:  hl.cmp slt
    // CHECK: hl.implicit_cast [[R:%[0-9]+]]  IntegralToBoolean : !hl.int -> !hl.bool
    return ((x ^ y) < 0);
}
