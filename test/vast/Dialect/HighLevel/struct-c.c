// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

// CHECK: hl.struct @"[[N:anonymous\[[0-9]+\]]]" : {
// CHECK:  hl.field @data : !hl.int
// CHECK: }
// CHECK: hl.var @named, <external> : !hl.lvalue<!hl.elaborated<!hl.record<@"[[N]]">>>
struct {
  int data;
} named;
