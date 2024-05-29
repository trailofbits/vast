// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

// CHECK: hl.var @p : !hl.lvalue<!hl.ptr<!hl.float>>
// CHECK: hl.var @pp : !hl.lvalue<!hl.ptr<!hl.ptr<!hl.float>>>
float *p, **pp; // p is a pointer to float
                // pp is a pointer to a pointer to float

// CHECK: hl.var @fp : !hl.lvalue<!hl.ptr<!hl.paren<!core.fn<(!hl.lvalue<!hl.int>) -> (!hl.int)>>>>
int (*fp)(int); // fp is a pointer to function with type int(int)

// CHECK: hl.var @pc : !hl.lvalue<!hl.ptr<!hl.int< const >>>
int n;
const int * pc = &n; // pc is a non-const pointer to a const int

// CHECK: hl.var @cp : !hl.lvalue<!hl.ptr<!hl.int,  const >>
int * const cp = &n; // cp is a const pointer to a non-const int

// CHECK: hl.var @pcp : !hl.lvalue<!hl.ptr<!hl.ptr<!hl.int,  const >>>
int * const * pcp = &cp; // non-const pointer to const pointer to non-const int

// CHECK: hl.var @np : !hl.lvalue<!hl.ptr<!hl.int>>
int *np = &n; // pointer to int
// CHECK: hl.var @npp : !hl.lvalue<!hl.ptr<!hl.ptr<!hl.int,  const >>>
int *const *npp = &np; // non-const pointer to const pointer to non-const int

int a[2];
// CHECK: hl.var @ap : !hl.lvalue<!hl.ptr<!hl.paren<!hl.array<2, !hl.int>>>>
int (*ap)[2] = &a; // pointer to array of int

struct S { int n; } s = {1};
// CHECK: hl.var @sp : !hl.lvalue<!hl.ptr<!hl.int>>
int* sp = &s.n; // pointer to the int that is a member of s
