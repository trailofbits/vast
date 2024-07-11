// RUN: %vast-front -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-front -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -
// REQUIRES: record-decl-typedef

// CHECK-DAG: hl.typedef "va_list"
#include <stdarg.h>
