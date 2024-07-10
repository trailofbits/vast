// RUN: %vast-front -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-front -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

// CHECK-DAG: hl.typedef "fenv_t"
// CHECK-DAG: hl.typedef "fexcept_t"
// CHECK-DAG: hl.func @feclearexcept
// CHECK-DAG: hl.func @fetestexcept
// CHECK-DAG: hl.func @feraiseexcept
// CHECK-DAG: hl.func @fegetexceptflag
// CHECK-DAG: hl.func @fesetexceptflag
// CHECK-DAG: hl.func @fegetround
// CHECK-DAG: hl.func @fesetround
// CHECK-DAG: hl.func @fegetenv
// CHECK-DAG: hl.func @fesetenv
// CHECK-DAG: hl.func @feholdexcept
// CHECK-DAG: hl.func @feupdateenv

#include <fenv.h>
