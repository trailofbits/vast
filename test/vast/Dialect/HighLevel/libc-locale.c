// RUN: %vast-front -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-front -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

// CHECK-DAG: hl.func @setlocale
// CHECK-DAG: hl.func @localeconv
// CHECK-DAG: hl.struct @lconv
#include <locale.h>
