// RUN: vast-cc --ccopts -xc --from-source %s | FileCheck %s
// RUN: vast-cc --ccopts -xc --from-source %s > %t && vast-opt %t | diff -B %t -

// CHECK: hl.var "a" : !hl.lvalue<!hl.ptr<!hl.float,  restrict >>
// CHECK: hl.var "b" : !hl.lvalue<!hl.ptr<!hl.float,  restrict >>
float * restrict a, * restrict b;

// CHECK: @f(!hl.lvalue<!hl.int>, !hl.lvalue<!hl.ptr<!hl.float,  restrict >>, !hl.lvalue<!hl.ptr<!hl.float,  const >>) -> !hl.void
void f(int n, float * restrict a, float * const b);

struct t {
   int n;
   // CHECK: hl.field "p" : !hl.ptr<!hl.float,  restrict >
   float * restrict p;
   // CHECK: hl.field "q" : !hl.ptr<!hl.float,  restrict >
   float * restrict q;
};

const struct t tv;
