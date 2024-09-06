// RUN: %vast-front -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-front -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

// CHECK-DAG: hl.func @difftime
// CHECK-DAG: hl.func @time
// CHECK-DAG: hl.func @clock
// CHECK-DAG: hl.func @timespec_get

// CHECK-DAG: hl.func @asctime
// CHECK-DAG: hl.func @ctime
// CHECK-DAG: hl.func @gmtime
// CHECK-DAG: hl.func @localtime

// CHECK-DAG: hl.struct @tm
// CHECK-DAG: hl.struct @timespec
// CHECK-DAG: hl.typedef @time_t
// CHECK-DAG: hl.typedef @clock_t

#include <time.h>
