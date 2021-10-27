// RUN: vast-cc --from-source %s | FileCheck %s

// CHECK: hl.var( i ): !hl.int
int i;

// CHECK: hl.var( u ): !hl<"unsigned int">
unsigned u;

// CHECK: hl.var( s ): !hl.int
signed s;

// CHECK: hl.var( ui ): !hl<"unsigned int">
unsigned int ui;

// CHECK: hl.var( us ): !hl<"unsigned short">
unsigned short us;

// CHECK: [[C1:%[0-9]+]] = hl.constant 0 : !hl.int
// CHECK-NEXT: hl.var( ci, [[C1]] ): !hl<"const int">
const int ci = 0;

// CHECK: [[C2:%[0-9]+]] = hl.constant 0 : !hl<"unsigned int">
// CHECK-NEXT: hl.var( cui, [[C2]] ): !hl<"const unsigned int">
const unsigned cui = 0U;

// CHECK: hl.var( vi ): !hl<"volatile int">
volatile int vi;

// CHECK: hl.var( vui ): !hl<"volatile unsigned int">
volatile unsigned vui;

// CHECK: [[C3:%[0-9]+]] = hl.constant 0 : !hl.int
// CHECK-NEXT: hl.var( cvi, [[C3]] ): !hl<"const volatile int">
const volatile int cvi = 0;

// CHECK: [[C4:%[0-9]+]] = hl.constant 0 : !hl<"unsigned int">
// CHECK-NEXT: hl.var( cvui, [[C4]] ): !hl<"const volatile unsigned int">
const volatile unsigned int cvui = 0U;

// CHECK: hl.var( b ): !hl.bool
bool b;

// CHECK: hl.var( vb ): !hl<"volatile bool">
volatile bool vb;

// CHECK: [[C5:%[0-9]+]] = hl.constant false : !hl.bool
// CHECK-NEXT: hl.var( cb, [[C5]] ): !hl<"const bool">
const bool cb = false;

// CHECK: [[C6:%[0-9]+]] = hl.constant false : !hl.bool
// CHECK-NEXT: hl.var( cvb, [[C6]] ): !hl<"const volatile bool">
const volatile bool cvb = false;
