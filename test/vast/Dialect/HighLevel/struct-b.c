// RUN: vast-cc --ccopts -xc --from-source %s | FileCheck %s
// RUN: vast-cc --ccopts -xc --from-source %s > %t && vast-opt %t | diff -B %t -

// CHECK: hl.struct "struct node" : {
// CHECK:  hl.field "data" : !hl.int
// CHECK:  hl.field "left" : !hl.ptr<!hl.named_type<"struct node">>
// CHECK:  hl.field "right" : !hl.ptr<!hl.named_type<"struct node">>
// CHECK: }
struct node {
  int data;
  struct node *left;
  struct node *right;
};
