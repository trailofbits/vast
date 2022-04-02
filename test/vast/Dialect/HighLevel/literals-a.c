// RUN: vast-cc --ccopts -xc --from-source %s | FileCheck %s
// RUN: vast-cc --ccopts -xc --from-source %s > %t && vast-opt %t | diff -B %t -

// CHECK: hl.var "li" : !hl.lvalue<!hl.int><const>
// CHECK: hl.constant.int 10 : !hl.int
const int li = 10;

// CHECK: hl.var "lui" : !hl.lvalue<!hl.int><unsigned const>
// CHECK: hl.constant.int 10 : !hl.int<unsigned>
const unsigned int lui = 10u;

// CHECK: hl.var "ll" : !hl.long<const>
// CHECK: hl.constant.int 10 : !hl.long
const long  ll = 10l;

// CHECK: hl.var "lf" : !hl.float<const>
// CHECK: hl.constant.float 5.000000e-01 : !hl.float
const float  lf = 0.5f;

// CHECK: hl.var "ld" : !hl.double<const>
// CHECK: hl.constant.float 5.000000e-01 : !hl.double
const double ld = 0.5;

// CHECK: hl.var "lc" : !hl.char<const>
// CHECK: hl.constant.int 97 : !hl.int
// CHECK: IntegralCast : !hl.int -> !hl.char
const char lc = 'a';

// CHECK: hl.var "null" : !hl.ptr<!hl.void>
// CHECK: hl.constant.int 0 : !hl.int
// CHECK: NullToPointer : !hl.int -> !hl.ptr<!hl.void>
const void *null = 0;

// CHECK: hl.var "lb" : !hl.bool<const>
// CHECK: hl.constant.int 1 : !hl.int
// CHECK: IntegralToBoolean : !hl.int -> !hl.bool
const _Bool lb = 1;
