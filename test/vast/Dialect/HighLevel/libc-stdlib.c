// RUN: %vast-front -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-front -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

// CHECK-DAG: hl.func @abort
// CHECK-DAG: hl.func @exit
// CHECK-DAG: hl.func @quick_exit
// CHECK-DAG: hl.func @_Exit
// CHECK-DAG: hl.func @atexit
// CHECK-DAG: hl.func @at_quick_exit

// CHECK-DAG: hl.func @system
// CHECK-DAG: hl.func @getenv
#include <stdlib.h>
