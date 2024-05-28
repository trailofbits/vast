// RUN: %vast-front -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-front -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -
#include <complex.h>

void fun() {
    float complex x = 3;
    float y = 3;
// CHECK: hl.fcmp {{.*}} : !hl.complex<!hl.float>, !hl.float
    int cmp1 = x == y;
// CHECK: hl.fcmp {{.*}} : !hl.float, !hl.complex<!hl.float>
    int cmp2 = y == x;
}
