// RUN: %vast-front -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-front -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

// CHECK-DAG: hl.func @isalnum
// CHECK-DAG: hl.func @isalpha
// CHECK-DAG: hl.func @islower
// CHECK-DAG: hl.func @isupper
// CHECK-DAG: hl.func @isdigit
// CHECK-DAG: hl.func @isxdigit
// CHECK-DAG: hl.func @iscntrl
// CHECK-DAG: hl.func @isgraph
// CHECK-DAG: hl.func @isspace
// CHECK-DAG: hl.func @isblank
// CHECK-DAG: hl.func @isprint
// CHECK-DAG: hl.func @ispunct

// CHECK-DAG: hl.func @tolower
// CHECK-DAG: hl.func @toupper

#include <ctype.h>