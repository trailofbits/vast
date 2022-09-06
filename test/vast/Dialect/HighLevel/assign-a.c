// RUN: vast-cc --ccopts -xc --from-source %s | FileCheck %s
// RUN: vast-cc --from-source %s > %t && vast-opt %t | diff -B %t -

int a, b;

void foo() {
    // CHECK: [[AG:%[0-9]+]] = hl.global.ref "a" : !hl.lvalue<!hl.int>
    // CHECK: [[A:%[0-9]+]] = hl.decl.ref [[AG]] : !hl.lvalue<!hl.int>
    // CHECK: [[BG:%[0-9]+]] = hl.global.ref "b" : !hl.lvalue<!hl.int>
    // CHECK: [[B1:%[0-9]+]] = hl.decl.ref [[BG]] : !hl.lvalue<!hl.int>
    // CHECK: [[B2:%[0-9]+]] = hl.pre.inc [[B1]] : !hl.lvalue<!hl.int>
    // CHECK: hl.assign [[B2]] to [[A]] : !hl.lvalue<!hl.int>, !hl.lvalue<!hl.int> -> !hl.lvalue<!hl.int>
    a = ++b;
}
