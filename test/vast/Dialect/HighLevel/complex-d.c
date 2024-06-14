// RUN: %vast-front -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-front -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

#include <complex.h>

void fun(void) {
    int complex x_i = 0;
    int complex y_i = 0;
    int z_i = 0;
    int b;
// CHECK: hl.assign.add {{.*}} : !hl.complex<!hl.int>, !hl.lvalue<!hl.complex<!hl.int>> -> !hl.complex<!hl.int>
    x_i += y_i;
// CHECK: hl.assign.add {{.*}} : !hl.complex<!hl.int>, !hl.lvalue<!hl.complex<!hl.int>> -> !hl.complex<!hl.int>
    x_i += z_i;
// CHECK: hl.cmp eq {{.*}} : !hl.complex<!hl.int>, !hl.complex<!hl.int> -> !hl.int
    b = x_i == y_i;
// CHECK: hl.cmp eq {{.*}} : !hl.complex<!hl.int>, !hl.complex<!hl.int> -> !hl.int
    b = x_i == z_i;
// CHECK: hl.cmp eq {{.*}} : !hl.complex<!hl.int>, !hl.complex<!hl.int> -> !hl.int
    b = z_i == x_i;


    unsigned int complex x_u = 0;
    unsigned int complex y_u = 0;
    unsigned int z_u = 0;
// CHECK: hl.assign.add {{.*}} : !hl.complex<!hl.int< unsigned >>, !hl.lvalue<!hl.complex<!hl.int< unsigned >>> -> !hl.complex<!hl.int< unsigned >>
    x_u += y_u;
// CHECK: hl.assign.add {{.*}} : !hl.complex<!hl.int< unsigned >>, !hl.lvalue<!hl.complex<!hl.int< unsigned >>> -> !hl.complex<!hl.int< unsigned >>

    x_u += z_u;
// CHECK: hl.cmp eq {{.*}} : !hl.complex<!hl.int< unsigned >>, !hl.complex<!hl.int< unsigned >> -> !hl.int
    b = x_u == y_u;
// CHECK: hl.cmp eq {{.*}} : !hl.complex<!hl.int< unsigned >>, !hl.complex<!hl.int< unsigned >> -> !hl.int
    b = x_u == z_u;
// CHECK: hl.cmp eq {{.*}} : !hl.complex<!hl.int< unsigned >>, !hl.complex<!hl.int< unsigned >> -> !hl.int
    b = z_u == x_u;

    float complex x_f = 0;
    float complex y_f = 0;
    float z_f = 0;
// CHECK: hl.assign.fadd {{.*}} : !hl.complex<!hl.float>, !hl.lvalue<!hl.complex<!hl.float>> -> !hl.complex<!hl.float>
    x_f += y_f;
// CHECK: hl.assign.fadd {{.*}} : !hl.float, !hl.lvalue<!hl.complex<!hl.float>> -> !hl.complex<!hl.float>
    x_f += z_f;
// CHECK: hl.fcmp oeq {{.*}} : !hl.complex<!hl.float>, !hl.complex<!hl.float> -> !hl.int
    b = x_f == y_f;
// CHECK: hl.fcmp oeq {{.*}} : !hl.complex<!hl.float>, !hl.float -> !hl.int
    b = x_f == z_f;
// CHECK: hl.fcmp oeq {{.*}} : !hl.float, !hl.complex<!hl.float> -> !hl.int
    b = z_f == x_f;
}
