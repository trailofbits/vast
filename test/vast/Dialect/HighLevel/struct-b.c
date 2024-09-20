// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

// CHECK: hl.struct @node : {
// CHECK:  hl.field @data : !hl.int
// CHECK:  hl.field @left : !hl.ptr<!hl.elaborated<!hl.record<@node>>>
// CHECK:  hl.field @right : !hl.ptr<!hl.elaborated<!hl.record<@node>>>
// CHECK: }
struct node {
  int data;
  struct node *left;
  struct node *right;
};
