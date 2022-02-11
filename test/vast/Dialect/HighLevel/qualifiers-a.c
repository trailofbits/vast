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

// CHECK: hl.var @ci : !hl.int<const> = {
// CHECK: [[C1:%[0-9]+]] = hl.constant.int 0 : !hl.int
// CHECK: hl.value.yield [[C1]]
const int ci = 0;

// CHECK: hl.var @cui : !hl.int<unsigned const> = {
// CHECK: [[C2:%[0-9]+]] = hl.constant.int 0 : !hl.int<unsigned>
// CHECK: hl.value.yield [[C2]]
const unsigned cui = 0U;

// CHECK: hl.var @vi : !hl.int<volatile>
volatile int vi;

// CHECK: hl.var @vui : !hl.int<unsigned volatile>
volatile unsigned vui;

// CHECK: hl.var @cvi : !hl.int<const volatile> = {
// CHECK: [[C3:%[0-9]+]] = hl.constant.int 0 : !hl.int
// CHECK: hl.value.yield [[C3]]
const volatile int cvi = 0;

// CHECK: hl.var @cvui : !hl.int<unsigned const volatile> = {
// CHECK: [[C4:%[0-9]+]] = hl.constant.int 0 : !hl.int<unsigned>
// CHECK: hl.value.yield [[C4]]
const volatile unsigned int cvui = 0U;

// CHECK: hl.var @b : !hl.bool
bool b;

// CHECK: hl.var @vb : !hl.bool<volatile>
volatile bool vb;

// CHECK: hl.var @cb : !hl.bool<const> = {
// CHECK: [[C5:%[0-9]+]] = hl.constant.int false : !hl.bool
// CHECK: hl.value.yield [[C5]]
const bool cb = false;

// CHECK: hl.var @cvb : !hl.bool<const volatile> = {
// CHECK: [[C6:%[0-9]+]] = hl.constant.int true : !hl.bool
// CHECK: hl.value.yield [[C6]]
const volatile bool cvb = true;
