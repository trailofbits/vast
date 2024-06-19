// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

// CHECK: !hl.lvalue<!hl.atomic<!hl.int>>
_Atomic int ai;

// CHECK: !hl.lvalue<!hl.ptr<!hl.atomic<!hl.int,  const >>>
_Atomic const int * paci;

// CHECK: !hl.lvalue<!hl.ptr<!hl.atomic<!hl.int>,  const >>
_Atomic int * const aipc;

// CHECK: !hl.lvalue<!hl.atomic<!hl.ptr<!hl.int< const >>>>
_Atomic(const int*) acip;

// CHECK: !hl.lvalue<!hl.atomic<!hl.int,  volatile >>
volatile _Atomic int vai;

// CHECK: !hl.lvalue<!hl.ptr<!hl.atomic<!hl.int>>>
_Atomic int * aip;

// CHECK: !hl.lvalue<!hl.ptr<!hl.atomic<!hl.int>,  restrict >>
_Atomic int *restrict aipr;
