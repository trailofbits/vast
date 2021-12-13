// RUN: vast-cc --ccopts -xc --from-source %s | FileCheck %s
// RUN: vast-cc --ccopts -xc --from-source %s > %t && vast-opt %t | diff -B %t -

// CHECK: hl.global @li : !hl.int<const>
// CHECK: hl.constant(#hl.int<10>) : !hl.int
const int   li = 10;

// CHECK: hl.global @lui : !hl.int<unsigned const>
// CHECK: hl.constant(#hl.int<10>) : !hl.int<unsigned>
const unsigned int lui = 10u;

// CHECK: hl.global @ll : !hl.long<const>
// CHECK: hl.constant(#hl.long<10>) : !hl.long
const long  ll = 10l;

// CHECK: hl.global @lf : !hl.float<const>
// CHECK: hl.constant(#hl.float<5.000000e-01>) : !hl.float
const float  lf = 0.5f;

// CHECK: hl.global @ld : !hl.double<const>
// CHECK: hl.constant(#hl.double<5.000000e-01>) : !hl.double
const double ld = 0.5;

// CHECK: hl.global @lc : !hl.char<const>
// CHECK: hl.constant(#hl.int<97>) : !hl.int
// CHECK: IntegralCast : !hl.int -> !hl.char
const char lc = 'a';

// CHECK: hl.global @null : !hl.ptr<!hl.void>
// CHECK: hl.constant(#hl.int<0>) : !hl.int
// CHECK: NullToPointer : !hl.int -> !hl.ptr<!hl.void>
const void *null = 0;

// CHECK: hl.global @lb : !hl.bool<const>
// CHECK: hl.constant(#hl.int<1>) : !hl.int
// CHECK: IntegralToBoolean : !hl.int -> !hl.bool
const _Bool lb = 1;
