// RUN: %vast-front -vast-emit-mlir=hl -std=c23 %s -o - | %file-check %s
// RUN: %vast-front -vast-emit-mlir=hl %s -o - | %file-check %s --check-prefix PRE-C23
// RUN: %vast-front -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

// CHECK-DAG: hl.func @mbrtoc8
// CHECK-DAG: hl.func @c8rtomb
// CHECK-DAG: hl.func @mbrtoc16
// CHECK-DAG: hl.func @c16rtomb
// CHECK-DAG: hl.func @mbrtoc32
// CHECK-DAG: hl.func @c32rtomb

// CHECK-DAG: hl.typedef "char8_t"
// CHECK-DAG: hl.typedef "char16_t"
// CHECK-DAG: hl.typedef "char32_t"

// PRE-C23-NOT: hl.typedef "char8_t"
#include <uchar.h>
