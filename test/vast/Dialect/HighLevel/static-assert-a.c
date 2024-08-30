// RUN: %vast-cc1 -std=c23 -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-cc1 -std=c23 -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

// CHECK: hl.static_assert failed : false
// hl.value.yield %{{[0-9]+}} : !hl.bool
// hl.const "msg" : !hl.lvalue<none>
static_assert(1>0, "msg");
// CHECK: hl.static_assert failed : false
// hl.value.yield %{{[0-9]+}} : !hl.bool
static_assert(1>0);
