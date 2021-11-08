// RUN: vast-cc --from-source %s | FileCheck %s
// RUN: vast-cc --from-source %s > %t && vast-opt %t | diff -B %t -

// CHECK: hl.var @i : !hl.int
int i;

// CHECK: hl.var @u : !hl.int<unsigned>
unsigned u;

// CHECK: hl.var @s : !hl.int
signed s;

// CHECK: hl.var @ui : !hl.int<unsigned>
unsigned int ui;

// CHECK: hl.var @us : !hl.short<unsigned>
unsigned short us;

// CHECK: [[C1:%[0-9]+]] = hl.constant 0 : !hl.int
// CHECK: [[C2:%[0-9]+]] =  hl.implicit_cast [[C1]] {{.*}} -> !hl.int<const>
// CHECK: hl.var @ci = [[C2]] : !hl.int<const>
const int ci = 0;

// CHECK: [[C3:%[0-9]+]] = hl.constant 0 : !hl.int<unsigned>
// CHECK: [[C4:%[0-9]+]] =  hl.implicit_cast [[C3]] {{.*}} -> !hl.int<unsigned const>
// CHECK: hl.var @cui = [[C4]] : !hl.int<unsigned const>
const unsigned cui = 0U;

// CHECK: hl.var @vi : !hl.int<volatile>
volatile int vi;

// CHECK: hl.var @vui : !hl.int<unsigned volatile>
volatile unsigned vui;

// CHECK: [[C5:%[0-9]+]] = hl.constant 0 : !hl.int
// CHECK: [[C6:%[0-9]+]] =  hl.implicit_cast [[C5]] {{.*}} -> !hl.int<const volatile>
// CHECK: hl.var @cvi = [[C6]] : !hl.int<const volatile>
const volatile int cvi = 0;

// CHECK: [[C7:%[0-9]+]] = hl.constant 0 : !hl.int<unsigned>
// CHECK: [[C8:%[0-9]+]] =  hl.implicit_cast [[C7]] {{.*}} -> !hl.int<unsigned const volatile>
// CHECK: hl.var @cvui = [[C8]] : !hl.int<unsigned const volatile>
const volatile unsigned int cvui = 0U;

// CHECK: hl.var @b : !hl.bool
bool b;

// CHECK: hl.var @vb : !hl.bool<volatile>
volatile bool vb;

// CHECK: [[C9:%[0-9]+]] = hl.constant false : !hl.bool
// CHECK: [[C10:%[0-9]+]] =  hl.implicit_cast [[C9]] {{.*}} -> !hl.bool<const>
// CHECK: hl.var @cb = [[C10]] : !hl.bool<const>
const bool cb = false;

// CHECK: [[C11:%[0-9]+]] = hl.constant true : !hl.bool
// CHECK: [[C12:%[0-9]+]] =  hl.implicit_cast [[C11]] {{.*}} -> !hl.bool<const volatile>
// CHECK: hl.var @cvb = [[C12]] : !hl.bool<const volatile>
const volatile bool cvb = true;
