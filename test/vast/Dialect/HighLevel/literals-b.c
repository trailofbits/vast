// // RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %file-check %s
// // RUN: %vast-cc1 -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -
struct S {
  int arr[2];
};

// CHECK: hl.compound_literal : !hl.lvalue
struct S *s = &(struct S){{[0] = 1, 1 + 1}};
