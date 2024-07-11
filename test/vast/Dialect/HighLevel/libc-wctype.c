// RUN: %vast-front -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-front -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

// CHECK-DAG: hl.func @iswalnum
// CHECK-DAG: hl.func @iswalpha
// CHECK-DAG: hl.func @iswlower
// CHECK-DAG: hl.func @iswupper
// CHECK-DAG: hl.func @iswdigit
// CHECK-DAG: hl.func @iswxdigit
// CHECK-DAG: hl.func @iswcntrl
// CHECK-DAG: hl.func @iswgraph
// CHECK-DAG: hl.func @iswspace
// CHECK-DAG: hl.func @iswblank
// CHECK-DAG: hl.func @iswprint
// CHECK-DAG: hl.func @iswpunct
// CHECK-DAG: hl.func @iswctype
// CHECK-DAG: hl.func @wctype

// CHECK-DAG: hl.func @towlower
// CHECK-DAG: hl.func @towupper
// CHECK-DAG: hl.func @towctrans
// CHECK-DAG: hl.func @wctrans

// CHECK-DAG: hl.typedef "wctrans_t"
// CHECK-DAG: hl.typedef "wctype_t"

#include <wctype.h>
