// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

// CHECK: @f {{.*}} (!hl.lvalue<!hl.int>, !hl.lvalue<!hl.int>, !hl.lvalue<!hl.decayed<!hl.ptr<!hl.array<?, !hl.float>,  restrict >>>, !hl.lvalue<!hl.decayed<!hl.ptr<!hl.array<?, !hl.float>,  restrict >>>)
void f(int m, int n, float a[restrict m][n], float b[restrict m][n]);
