// // RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %file-check %s
// // RUN: %vast-cc1 -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

// CHECK: hl.var @li, <external> : !hl.lvalue<!hl.int< const >>
// CHECK: hl.const #core.integer<10> : !hl.int
const int li = 10;

// CHECK: hl.var @lui, <external> : !hl.lvalue<!hl.int< unsigned, const >>
// CHECK: hl.const #core.integer<10> : !hl.int< unsigned >
const unsigned int lui = 10u;

// CHECK: hl.var @ll, <external> : !hl.lvalue<!hl.long< const >>
// CHECK: hl.const #core.integer<10> : !hl.long
const long  ll = 10l;

// CHECK: hl.var @lf, <external> : !hl.lvalue<!hl.float< const >>
// CHECK: hl.const #core.float<5.000000e-01> : !hl.float
const float  lf = 0.5f;

// CHECK: hl.var @ld, <external> : !hl.lvalue<!hl.double< const >>
// CHECK: hl.const #core.float<5.000000e-01> : !hl.double
const double ld = 0.5;

// CHECK: hl.var @lc, <external> : !hl.lvalue<!hl.char< const >>
// CHECK: hl.const #core.integer<97> : !hl.int
// CHECK: IntegralCast : !hl.int -> !hl.char
const char lc = 'a';

// CHECK: hl.var @null, <external> : !hl.lvalue<!hl.ptr<!hl.void< const >>>
// CHECK: hl.const #core.integer<0> : !hl.int
// CHECK: NullToPointer : !hl.int -> !hl.ptr<!hl.void< const >>
const void *null = 0;

// CHECK: hl.var @lb, <external> : !hl.lvalue<!hl.bool< const >>
// CHECK: hl.const #core.integer<1> : !hl.int
// CHECK: IntegralToBoolean : !hl.int -> !hl.bool
const _Bool lb = 1;

#define SCHAR_MIN	(-128)
#define SCHAR_MAX	127

// CHECK: hl.var @scmin, <external> : !hl.lvalue<!hl.char< const >>
// CHECK:   hl.const #core.integer<128> : !hl.int
// CHECK:   hl.minus
const char scmin = SCHAR_MIN;

// CHECK: hl.var @scmax, <external> : !hl.lvalue<!hl.char< const >>
// CHECK:   hl.const #core.integer<127> : !hl.int
const char scmax = SCHAR_MAX;

#define UCHAR_MAX	255

// CHECK: hl.var @ucmax, <external> : !hl.lvalue<!hl.char< unsigned, const >>
// CHECK:   hl.const #core.integer<255> : !hl.int
const unsigned char ucmax = UCHAR_MAX;
