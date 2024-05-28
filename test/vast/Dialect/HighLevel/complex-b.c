// RUN: %vast-front -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-front -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -
#include <complex.h>

void fun() {
    double complex x_d = 3;
    double y_d = 3;
    float complex x_f = 3;
    float y_f = 3;
// CHECK: hl.fcmp {{.*}} : !hl.complex<!hl.double>, !hl.double
    int cmp1 = x_d == y_d;
// CHECK: hl.fcmp {{.*}} : !hl.double, !hl.complex<!hl.double>
    int cmp2 = y_d == x_d;
// CHECK: hl.fcmp {{.*}} : !hl.complex<!hl.float>, !hl.float
    int cmp3 = x_f == y_f;
// CHECK: hl.fcmp {{.*}} : !hl.float, !hl.complex<!hl.float>
    int cmp4 = y_f == x_f;
// CHECK: hl.fcmp {{.*}} : !hl.complex<!hl.double>, !hl.double
    int cmp5 = x_d == y_f;
// CHECK: hl.fcmp {{.*}} : !hl.double, !hl.complex<!hl.double>
    int cmp6 = y_f == x_d;
// CHECK: hl.fcmp {{.*}} : !hl.complex<!hl.double>, !hl.double
    int cmp7 = x_f == y_d;
// CHECK: hl.fcmp {{.*}} : !hl.double, !hl.complex<!hl.double>
    int cmp8 = y_d == x_f;
}
