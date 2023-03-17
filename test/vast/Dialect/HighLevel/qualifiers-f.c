// RUN: vast-cc --ccopts -xc --from-source %s | FileCheck %s
// RUN: vast-cc --ccopts -xc --from-source %s > %t && vast-opt %t | diff -B %t -

// CHECK: @f (!hl.lvalue<!hl.int>, !hl.lvalue<!hl.int>, !hl.lvalue<!hl.decayed<!hl.ptr<!hl.array<?, !hl.float>,  restrict >>>, !hl.lvalue<!hl.decayed<!hl.ptr<!hl.array<?, !hl.float>,  restrict >>>) -> ()
void f(int m, int n, float a[restrict m][n], float b[restrict m][n]);
