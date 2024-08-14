// RUN: %vast-front -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-front -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

// CHECK-DAG: hl.typedef @int8_t
// CHECK-DAG: hl.typedef @int16_t
// CHECK-DAG: hl.typedef @int32_t
// CHECK-DAG: hl.typedef @int64_t

// CHECK-DAG: hl.typedef @int_fast8_t
// CHECK-DAG: hl.typedef @int_fast16_t
// CHECK-DAG: hl.typedef @int_fast32_t
// CHECK-DAG: hl.typedef @int_fast64_t

// CHECK-DAG: hl.typedef @int_least8_t
// CHECK-DAG: hl.typedef @int_least16_t
// CHECK-DAG: hl.typedef @int_least32_t
// CHECK-DAG: hl.typedef @int_least64_t

// CHECK-DAG: hl.typedef @intmax_t

// CHECK-DAG: hl.typedef @intptr_t

// CHECK-DAG: hl.typedef @uint8_t
// CHECK-DAG: hl.typedef @uint16_t
// CHECK-DAG: hl.typedef @uint32_t
// CHECK-DAG: hl.typedef @uint64_t

// CHECK-DAG: hl.typedef @uint_fast8_t
// CHECK-DAG: hl.typedef @uint_fast16_t
// CHECK-DAG: hl.typedef @uint_fast32_t
// CHECK-DAG: hl.typedef @uint_fast64_t

// CHECK-DAG: hl.typedef @uint_least8_t
// CHECK-DAG: hl.typedef @uint_least16_t
// CHECK-DAG: hl.typedef @uint_least32_t
// CHECK-DAG: hl.typedef @uint_least64_t

// CHECK-DAG: hl.typedef @uintmax_t

// CHECK-DAG: hl.typedef @uintptr_t

// CHECK-DAG: hl.func @imaxabs
// CHECK-DAG: hl.func @imaxdiv
// CHECK-DAG: hl.func @strtoimax
// CHECK-DAG: hl.func @strtoumax
// CHECK-DAG: hl.func @wcstoimax
// CHECK-DAG: hl.func @wcstoumax

#include <inttypes.h>
