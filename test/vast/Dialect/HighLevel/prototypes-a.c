// RUN: vast-cc --ccopts -xc --from-source %s | FileCheck %s
// RUN: vast-cc --ccopts -xc --from-source %s > %t && vast-opt %t | diff -B %t -

short b(void);
// CHECK: func external @b () -> !hl.short

int c(void);
// CHECK: func external @c () -> !hl.int

long d(void);
// CHECK: func external @d () -> !hl.long

long long e(void);
// CHECK: func external @e () -> !hl.longlong

void f(void);
// CHECK: func external @f ()

void g(int);
// CHECK: func external @g (!hl.lvalue<!hl.int>)

void h(int, int);
// CHECK: func external @h (!hl.lvalue<!hl.int>, !hl.lvalue<!hl.int>)
