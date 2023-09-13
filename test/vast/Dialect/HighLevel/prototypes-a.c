// RUN: %vast-cc --ccopts -xc --from-source %s | %file-check %s
// RUN: %vast-cc --ccopts -xc --from-source %s > %t && %vast-opt %t | diff -B %t -

short b(void);
// CHECK: func @b () -> !hl.short

int c(void);
// CHECK: func @c () -> !hl.int

long d(void);
// CHECK: func @d () -> !hl.long

long long e(void);
// CHECK: func @e () -> !hl.longlong

void f(void);
// CHECK: func @f ()

void g(int);
// CHECK: func @g (!hl.lvalue<!hl.int>)

void h(int, int);
// CHECK: func @h (!hl.lvalue<!hl.int>, !hl.lvalue<!hl.int>)
