// RUN: %vast-front -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-front -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

#include <complex.h>

void fun(void) {
// CHECK: @x : !hl.lvalue<!hl.complex<!hl.int>>
    int complex x = I;
    int z = 3;

// CHECK: hl.add {{.*}} : (!hl.complex<!hl.int>, !hl.complex<!hl.int>)
    int complex y = x + z;
// CHEKC: hl.real {{.*}} : {{.*}}!hl.complex<!hl.int>{{.*}} -> !hl.int
    int u = __real__ y;
// CHEKC: hl.imag {{.*}} : {{.*}}!hl.complex<!hl.int>{{.*}} -> !hl.int
    int v = __imag__ y;
// CHEKC: hl.real {{.*}} : {{.*}}!hl.complex<!hl.int>{{.*}} -> !hl.lvalue<!hl.int>
    __real__ y = 5;
// CHEKC: hl.imag {{.*}} : {{.*}}!hl.complex<!hl.int>{{.*}} -> !hl.lvalue<!hl.int>
    __imag__ y = 6;
}
