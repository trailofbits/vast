// RUN: %vast-front -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-front -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

// CHECK-DAG: hl.func @setjmp
// CHECK-DAG: hl.func @longjmp
// CHECK-DAG: hl.typedef @jmp_buf
#include <setjmp.h>
