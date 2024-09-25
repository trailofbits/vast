// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

// CHECK: hl.var @p, <external> : !hl.lvalue<!hl.ptr<!hl.void>>
void *p;

// CHECK: hl.var @i, <external> : !hl.lvalue<!hl.ptr<!hl.int>>
int *i;

// CHECK: hl.var @u, <external> : !hl.lvalue<!hl.ptr<!hl.int< unsigned >>>
unsigned *u;

// CHECK: hl.var @s, <external> : !hl.lvalue<!hl.ptr<!hl.int>>
signed *s;

// CHECK: hl.var @ui, <external> : !hl.lvalue<!hl.ptr<!hl.int< unsigned >>>
unsigned int *ui;

// CHECK: hl.var @us, <external> : !hl.lvalue<!hl.ptr<!hl.short< unsigned >>>
unsigned short *us;

// CHECK: hl.var @ci, <external> : !hl.lvalue<!hl.ptr<!hl.int< const >>>
const int *ci = 0;

// CHECK: hl.var @cui, <external> : !hl.lvalue<!hl.ptr<!hl.int< unsigned, const >>>
const unsigned *cui = 0;

// CHECK: hl.var @vi, <external> : !hl.lvalue<!hl.ptr<!hl.int< volatile >>>
volatile int *vi;

// CHECK: hl.var @vui, <external> : !hl.lvalue<!hl.ptr<!hl.int< unsigned, volatile >>>
volatile unsigned *vui;

// CHECK: hl.var @cvi, <external> : !hl.lvalue<!hl.ptr<!hl.int< const, volatile >>>
const volatile int *cvi = 0;

// CHECK: hl.var @cvui, <external> : !hl.lvalue<!hl.ptr<!hl.int< unsigned, const, volatile >>>
const volatile unsigned int *cvui = 0U;

// CHECK: hl.var @b, <external> : !hl.lvalue<!hl.ptr<!hl.bool>>
bool *b;

// CHECK: hl.var @vb, <external> : !hl.lvalue<!hl.ptr<!hl.bool< volatile >>>
volatile bool *vb;

// CHECK: hl.var @cb, <external> : !hl.lvalue<!hl.ptr<!hl.bool< const >>>
const bool *cb = 0;

// CHECK: hl.var @cvb, <external> : !hl.lvalue<!hl.ptr<!hl.bool< const, volatile >>>
const volatile bool *cvb = 0;
