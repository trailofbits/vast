// RUN: vast-cc --from-source %s | FileCheck %s

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
// CHECK-NEXT: hl.var @ci = [[C1]] : !hl.int<const>
const int ci = 0;

// CHECK: [[C2:%[0-9]+]] = hl.constant 0 : !hl.int<unsigned>
// CHECK-NEXT: hl.var @cui = [[C2]] : !hl.int<unsigned const>
const unsigned cui = 0U;

// CHECK: hl.var @vi : !hl.int<volatile>
volatile int vi;

// CHECK: hl.var @vui : !hl.int<unsigned volatile>
volatile unsigned vui;

// CHECK: [[C3:%[0-9]+]] = hl.constant 0 : !hl.int
// CHECK-NEXT: hl.var @cvi = [[C3]] : !hl.int<const volatile>
const volatile int cvi = 0;

// CHECK: [[C4:%[0-9]+]] = hl.constant 0 : !hl.int<unsigned>
// CHECK-NEXT: hl.var @cvui = [[C4]] : !hl.int<unsigned const volatile>
const volatile unsigned int cvui = 0U;

// CHECK: hl.var @b : !hl.bool
bool b;

// CHECK: hl.var @vb : !hl.bool<volatile>
volatile bool vb;

// CHECK: [[C5:%[0-9]+]] = hl.constant false : !hl.bool
// CHECK-NEXT: hl.var @cb = [[C5]] : !hl.bool<const>
const bool cb = false;

// CHECK: [[C6:%[0-9]+]] = hl.constant false : !hl.bool
// CHECK-NEXT: hl.var @cvb = [[C6]] : !hl.bool<const volatile>
const volatile bool cvb = false;
