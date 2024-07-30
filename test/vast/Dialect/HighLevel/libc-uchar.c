// RUN: %vast-front -vast-emit-mlir=hl %s -o - | %file-check %s --check-prefixes PRE-C23,CHECK
// RUN: %vast-front -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -
// UNSUPPORTED: system-darwin

// C23-DAG: hl.func @mbrtoc8
// C23-DAG: hl.func @c8rtomb
// CHECK-DAG: hl.func @mbrtoc16
// CHECK-DAG: hl.func @c16rtomb
// CHECK-DAG: hl.func @mbrtoc32
// CHECK-DAG: hl.func @c32rtomb

// C23-DAG: hl.typedef "char8_t"
// CHECK-DAG: hl.typedef "char16_t"
// CHECK-DAG: hl.typedef "char32_t"

// PRE-C23-NOT: hl.typedef "char8_t"
// PRE-C23-NOT: hl.func @mbrtoc8
// PRE-C23-NOT: hl.func @c8rtomb
#include <uchar.h>
