// RUN: %vast-front -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-front -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

// CHECK-DAG: hl.func @strcpy
// CHECK-DAG: hl.func @strncpy
// CHECK-DAG: hl.func @strcat
// CHECK-DAG: hl.func @strncat
// CHECK-DAG: hl.func @strxfrm
// CHECK-DAG: hl.func @strdup
// CHECK-DAG: hl.func @strndup

// CHECK-DAG: hl.func @strlen
// CHECK-DAG: hl.func @strcmp
// CHECK-DAG: hl.func @strncmp
// CHECK-DAG: hl.func @strcoll
// CHECK-DAG: hl.func @strrchr
// CHECK-DAG: hl.func @strspn
// CHECK-DAG: hl.func @strcspn
// CHECK-DAG: hl.func @strpbrk
// CHECK-DAG: hl.func @strstr
// CHECK-DAG: hl.func @strtok

// CHECK-DAG: hl.func @memchr
// CHECK-DAG: hl.func @memcmp
// CHECK-DAG: hl.func @memset
// CHECK-DAG: hl.func @memcpy

// CHECK-DAG: hl.func @strerror
#include <string.h>
