// RUN: vast-cc --from-source %s | FileCheck %s
// RUN: vast-cc --from-source %s > %t && vast-opt %t | diff -B %t -

// CHECK: hl.func external @f ([[ARG:%arg[0-9]+]]: !hl.lvalue<!hl.decayed<!hl.ptr<!hl.int>>>) -> !hl.void
void f(int i[]) {
    // CHECK: hl.subscript [[A:%[0-9]+]] at [[[D:%[0-9]+]] : !hl.int] : !hl.decayed<!hl.ptr<!hl.int>> -> !hl.lvalue<!hl.int>
    i[1] = 0;
}
