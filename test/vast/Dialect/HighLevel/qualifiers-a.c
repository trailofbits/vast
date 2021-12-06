// RUN: vast-cc --from-source %s | FileCheck %s
// RUN: vast-cc --from-source %s > %t && vast-opt %t | diff -B %t -

// CHECK: hl.global @i : !hl.int
int i;

// CHECK: hl.global @u : !hl.int<unsigned>
unsigned u;

// CHECK: hl.global @s : !hl.int
signed s;

// CHECK: hl.global @ui : !hl.int<unsigned>
unsigned int ui;

// CHECK: hl.global @us : !hl.short<unsigned>
unsigned short us;

// CHECK: hl.global @ci : !hl.int<const> = {
// CHECK: [[C1:%[0-9]+]] = hl.constant 0 : !hl.int
// CHECK: hl.value.yield [[C1]]
const int ci = 0;

// CHECK: hl.global @cui : !hl.int<unsigned const> = {
// CHECK: [[C2:%[0-9]+]] = hl.constant 0 : !hl.int<unsigned>
// CHECK: hl.value.yield [[C2]]
const unsigned cui = 0U;

// CHECK: hl.global @vi : !hl.int<volatile>
volatile int vi;

// CHECK: hl.global @vui : !hl.int<unsigned volatile>
volatile unsigned vui;

// CHECK: hl.global @cvi : !hl.int<const volatile> = {
// CHECK: [[C3:%[0-9]+]] = hl.constant 0 : !hl.int
// CHECK: hl.value.yield [[C3]]
const volatile int cvi = 0;

// CHECK: hl.global @cvui : !hl.int<unsigned const volatile> = {
// CHECK: [[C4:%[0-9]+]] = hl.constant 0 : !hl.int<unsigned>
// CHECK: hl.value.yield [[C4]]
const volatile unsigned int cvui = 0U;

// CHECK: hl.global @b : !hl.bool
bool b;

// CHECK: hl.global @vb : !hl.bool<volatile>
volatile bool vb;

// CHECK: hl.global @cb : !hl.bool<const> = {
// CHECK: [[C5:%[0-9]+]] = hl.constant false : !hl.bool
// CHECK: hl.value.yield [[C5]]
const bool cb = false;

// CHECK: hl.global @cvb : !hl.bool<const volatile> = {
// CHECK: [[C6:%[0-9]+]] = hl.constant true : !hl.bool
// CHECK: hl.value.yield [[C6]]
const volatile bool cvb = true;
