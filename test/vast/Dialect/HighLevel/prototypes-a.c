// RUN: vast-cc --from-source %s | FileCheck %s
// RUN: vast-cc --from-source %s > %t && vast-opt %t | diff -B %t -

short b(void);
// CHECK: func @b() -> !hl.short

int c(void);
// CHECK: func @c() -> !hl.int

long d(void);
// CHECK: func @d() -> !hl.long

long long e(void);
// CHECK: func @e() -> !hl.longlong

void f(void);
// CHECK: func @f() -> !hl.void

void g(int);
// CHECK: func @g(!hl.int) -> !hl.void

void h(int, int);
// CHECK: func @h(!hl.int, !hl.int) -> !hl.void
