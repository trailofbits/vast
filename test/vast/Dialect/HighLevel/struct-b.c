// RUN: %vast-cc --ccopts -xc --from-source %s | %file-check %s
// RUN: %vast-cc --ccopts -xc --from-source %s > %t && %vast-opt %t | diff -B %t -

// CHECK: hl.struct "node" : {
// CHECK:  hl.field "data" : !hl.int
// CHECK:  hl.field "left" : !hl.ptr<!hl.elaborated<!hl.record<"node">>>
// CHECK:  hl.field "right" : !hl.ptr<!hl.elaborated<!hl.record<"node">>>
// CHECK: }
struct node {
  int data;
  struct node *left;
  struct node *right;
};
