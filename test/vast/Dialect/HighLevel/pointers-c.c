// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

// CHECK: hl.var @p : !hl.lvalue<!hl.ptr<!hl.int>>
// CHECK:   ArrayToPointerDecay : !hl.lvalue<!hl.array<2, !hl.int>> -> !hl.ptr<!hl.int>
int a[2];
int *p = a; // pointer to a[0]

// CHECK: hl.var @row : !hl.lvalue<!hl.ptr<!hl.paren<!hl.array<3, !hl.int>>>>
// CHECK:   ArrayToPointerDecay : !hl.lvalue<!hl.array<3, !hl.array<3, !hl.int>>> -> !hl.ptr<!hl.array<3, !hl.int>>
int b[3][3];
int (*row)[3] = b; // pointer to b[0]
