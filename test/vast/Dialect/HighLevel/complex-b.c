// RUN: %vast-front -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-front -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -
#include <complex.h>

void fun() {
    double complex x = 3;
    double y = 3;
// CHECK: hl.fcmp {{.*}} : !hl.complex<!hl.double>, !hl.double
    int cmp = x == y;
}
