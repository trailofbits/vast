// RUN: %vast-front -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-front -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

// CHECK-DAG: hl.func @wcscpy
// CHECK-DAG: hl.func @wcsncpy
// CHECK-DAG: hl.func @wcscat
// CHECK-DAG: hl.func @wcsncat
// CHECK-DAG: hl.func @wcsxfrm

// CHECK-DAG: hl.func @wcslen
// CHECK-DAG: hl.func @wcscmp
// CHECK-DAG: hl.func @wcsncmp
// CHECK-DAG: hl.func @wcscoll
// CHECK-DAG: hl.func @wcschr
// CHECK-DAG: hl.func @wcsrchr
// CHECK-DAG: hl.func @wcsspn
// CHECK-DAG: hl.func @wcscspn
// CHECK-DAG: hl.func @wcspbrk
// CHECK-DAG: hl.func @wcsstr
// CHECK-DAG: hl.func @wcstok

// CHECK-DAG: hl.func @wmemcpy
// CHECK-DAG: hl.func @wmemmove
// CHECK-DAG: hl.func @wmemcmp
// CHECK-DAG: hl.func @wmemchr
// CHECK-DAG: hl.func @wmemset

// CHECK-DAG: hl.typedef "wchar_t"
// CHECK-DAG: hl.typedef "wint_t"
#include <wchar.h>
