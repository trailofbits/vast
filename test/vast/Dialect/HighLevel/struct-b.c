// RUN: vast-cc --ccopts -xc --from-source %s | FileCheck %s
// RUN: vast-cc --ccopts -xc --from-source %s > %t && vast-opt %t | diff -B %t -

// CHECK: hl.type.decl @node
// CHECK: hl.typedef @node : !hl.record<data : !hl.int, left : !hl.ptr<!hl.named_type<@node>>, right : !hl.ptr<!hl.named_type<@node>>>
struct node {
  int data;
  struct node *left;
  struct node *right;
};
