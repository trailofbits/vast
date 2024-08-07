// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

// CHECK: hl.var @p : !hl.lvalue<!hl.ptr<!hl.void>>
void *p;

// CHECK: hl.var @i : !hl.lvalue<!hl.ptr<!hl.int>>
int *i;

// CHECK: hl.var @u : !hl.lvalue<!hl.ptr<!hl.int< unsigned >>>
unsigned *u;

// CHECK: hl.var @s : !hl.lvalue<!hl.ptr<!hl.int>>
signed *s;

// CHECK: hl.var @ui : !hl.lvalue<!hl.ptr<!hl.int< unsigned >>>
unsigned int *ui;

// CHECK: hl.var @us : !hl.lvalue<!hl.ptr<!hl.short< unsigned >>>
unsigned short *us;

// CHECK: hl.var @ci : !hl.lvalue<!hl.ptr<!hl.int< const >>>
const int *ci = 0;

// CHECK: hl.var @cui : !hl.lvalue<!hl.ptr<!hl.int< unsigned, const >>>
const unsigned *cui = 0;

// CHECK: hl.var @vi : !hl.lvalue<!hl.ptr<!hl.int< volatile >>>
volatile int *vi;

// CHECK: hl.var @vui : !hl.lvalue<!hl.ptr<!hl.int< unsigned, volatile >>>
volatile unsigned *vui;

// CHECK: hl.var @cvi : !hl.lvalue<!hl.ptr<!hl.int< const, volatile >>>
const volatile int *cvi = 0;

// CHECK: hl.var @cvui : !hl.lvalue<!hl.ptr<!hl.int< unsigned, const, volatile >>>
const volatile unsigned int *cvui = 0U;

// CHECK: hl.var @b : !hl.lvalue<!hl.ptr<!hl.bool>>
bool *b;

// CHECK: hl.var @vb : !hl.lvalue<!hl.ptr<!hl.bool< volatile >>>
volatile bool *vb;

// CHECK: hl.var @cb : !hl.lvalue<!hl.ptr<!hl.bool< const >>>
const bool *cb = 0;

// CHECK: hl.var @cvb : !hl.lvalue<!hl.ptr<!hl.bool< const, volatile >>>
const volatile bool *cvb = 0;
