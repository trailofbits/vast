// RUN: %vast-front -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-front -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

// CHECK-DAG: hl.typedef @size_t
// CHECK-DAG: hl.typedef @ptrdiff_t
// CHECK-DAG: hl.typedef @max_align_t
#include <stddef.h>
