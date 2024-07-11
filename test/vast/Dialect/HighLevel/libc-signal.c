// RUN: %vast-front -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-front -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -
// REQUIRES: record-decl-typedef

// CHECK-DAG: hl.func @signal
// CHECK-DAG: hl.func @raise
// CHECK-DAG: hl.typedef "sig_atomic_t"
#include <signal.h>
