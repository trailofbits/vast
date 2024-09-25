// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

// CHECK: hl.var @i, <external> : !hl.lvalue<!hl.int>
// CHECK:   hl.const #core.integer<0> : !hl.int
int i = 0;

// CHECK: hl.var @ui, <external> : !hl.lvalue<!hl.int< unsigned >>
// CHECK:   hl.const #core.integer<0> : !hl.int< unsigned >
unsigned int ui = 0u;

// CHECK: hl.var @lli, <external> : !hl.lvalue<!hl.longlong>
// CHECK:   hl.const #core.integer<0> : !hl.longlong
long long int lli = 0ll;

// CHECK: hl.var @f, <external> : !hl.lvalue<!hl.float>
// CHECK:   hl.const #core.float<0.000000e+00> : !hl.float
float f = 0.f;

// CHECK: hl.var @d, <external> : !hl.lvalue<!hl.double>
// CHECK:   hl.const #core.float<0.000000e+00> : !hl.double
double d = 0.0;

// CHECK: hl.var @str, <external> : !hl.lvalue<!hl.ptr<!hl.char< const >>>
// CHECK: hl.const "hello" : !hl.lvalue<!hl.array<6, !hl.char>>
const char *str = "hello";

// CHECK: hl.var @arr, <external> : !hl.lvalue<!hl.array<3, !hl.int< const >>>
// CHECK:   hl.const #core.integer<1> : !hl.int
// CHECK:   hl.const #core.integer<2> : !hl.int
// CHECK:   hl.const #core.integer<3> : !hl.int
const int arr[] = { 1, 2, 3 };
