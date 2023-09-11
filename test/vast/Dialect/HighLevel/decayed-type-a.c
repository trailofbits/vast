// RUN: %vast-cc --ccopts -xc --from-source %s | FileCheck %s
// RUN: %vast-cc --ccopts -xc --from-source %s > %t && %vast-opt %t | diff -B %t -

// CHECK: hl.func @f ([[ARG:%arg[0-9]+]]: !hl.lvalue<!hl.decayed<!hl.ptr<!hl.int>>>)
void f(int i[]) {
    // CHECK: hl.subscript [[A:%[0-9]+]] at [[[D:%[0-9]+]] : !hl.int] : !hl.decayed<!hl.ptr<!hl.int>> -> !hl.lvalue<!hl.int>
    i[1] = 0;
}
