// RUN: %vast-front -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-front -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

#include <complex.h>

void fun(void) {
    int complex x_i = 0;
    int complex y_i = 0;
    int z_i = 0;
// CHECK: hl.assign.add {{.*}} : !hl.complex<!hl.int>, !hl.lvalue<!hl.complex<!hl.int>> -> !hl.complex<!hl.int>
    x_i += y_i;
// CHECK: hl.assign.add {{.*}} : !hl.complex<!hl.int>, !hl.lvalue<!hl.complex<!hl.int>> -> !hl.complex<!hl.int>
    x_i += z_i;
    unsigned int complex x_u = 0;
    unsigned int complex y_u = 0;
    unsigned int z_u = 0;
// CHECK: hl.assign.add {{.*}} : !hl.complex<!hl.int< unsigned >>, !hl.lvalue<!hl.complex<!hl.int< unsigned >>> -> !hl.complex<!hl.int< unsigned >>
    x_u += y_u;
// CHECK: hl.assign.add {{.*}} : !hl.complex<!hl.int< unsigned >>, !hl.lvalue<!hl.complex<!hl.int< unsigned >>> -> !hl.complex<!hl.int< unsigned >>

    x_u += z_u;
    float complex x_f = 0;
    float complex y_f = 0;
    float z_f = 0;
// CHECK: hl.assign.fadd {{.*}} : !hl.complex<!hl.float>, !hl.lvalue<!hl.complex<!hl.float>> -> !hl.complex<!hl.float>
    x_f += y_f;
// CHECK: hl.assign.fadd {{.*}} : !hl.float, !hl.lvalue<!hl.complex<!hl.float>> -> !hl.complex<!hl.float>
    x_f += z_f;
}
