// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

// CHECK: hl.var @fp : !hl.lvalue<!hl.ptr<!hl.paren<!core.fn<(!hl.lvalue<!hl.int>) -> (!hl.int)>>>>
int (*fp) (int);

// CHECK: hl.var @cfp : !hl.lvalue<!hl.ptr<!hl.paren<!core.fn<(!hl.lvalue<!hl.int>) -> (!hl.int< const >)>>>>
const int (*cfp)(int);

// CHECK: hl.var @fpc : !hl.lvalue<!hl.ptr<!hl.paren<!core.fn<(!hl.lvalue<!hl.int< const >>) -> (!hl.int)>>>>
int (*fpc)(const int);
