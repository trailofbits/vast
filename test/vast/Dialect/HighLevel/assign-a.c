// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

int a, b;

void foo() {
    // CHECK: [[A:%[0-9]+]] = hl.ref @a : !hl.lvalue<!hl.int>
    // CHECK: [[B:%[0-9]+]] = hl.ref @b : !hl.lvalue<!hl.int>
    // CHECK: [[B2:%[0-9]+]] = hl.pre.inc [[B]] : !hl.lvalue<!hl.int> -> !hl.int
    // CHECK: hl.assign [[B2]] to [[A]] : !hl.int, !hl.lvalue<!hl.int> -> !hl.int
    a = ++b;
}
