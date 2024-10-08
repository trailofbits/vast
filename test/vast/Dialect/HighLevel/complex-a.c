// RUN: %vast-front -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-front -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -
#include <complex.h>

void fun() {
// CHECK: @x : !hl.lvalue<!hl.complex<!hl.double>>
// CHECK: hl.const.init : !hl.complex<!hl.float>
    double complex x = I;
// CHECK: hl.fadd {{.*}} : (!hl.complex<!hl.double>, !hl.double)
    double complex y = x + 3;
    double c = 3;
// CHECK: hl.fadd {{.*}} : (!hl.complex<!hl.double>, !hl.double)
    double complex z = x + c;
// CHEKC: hl.real {{.*}} : !hl.lvalue<!hl.complex<!hl.double>> -> !hl.double
    double a = __real__ y;
// CHEKC: hl.imag {{.*}} : !hl.lvalue<!hl.complex<!hl.double>> -> !hl.double
    double b = __imag__ y;

// CHEKC: hl.real {{.*}} : !hl.lvalue<!hl.complex<!hl.double>> -> !hl.lvalue<!hl.double>
// CHEKC: hl.imag {{.*}} : !hl.lvalue<!hl.complex<!hl.double>> -> !hl.lvalue<!hl.double>
    __real__ x = 1;
    __imag__ x = 1;
}
