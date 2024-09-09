// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

struct S1 {
    // CHECK: hl.field @a : !hl.int< unsigned >
    // CHECK: hl.field @b bw 3 : !hl.int< unsigned >
    unsigned int a, b : 3;
};

struct S2 {
    // CHECK: hl.field @b1 bw 5 : !hl.int< unsigned >
    // CHECK: hl.field @"anonymous[{{[0-9]+}}]" bw 11 : !hl.int< unsigned >
    // CHECK: hl.field @b2 bw 6 : !hl.int< unsigned >
    // CHECK: hl.field @b3 bw 2 : !hl.int< unsigned >
    unsigned b1 : 5, : 11, b2 : 6, b3 : 2;
};

struct S3 {
    // CHECK: hl.field @b1 bw 5 : !hl.int< unsigned >
    // CHECK: hl.field @"anonymous[{{[0-9]+}}]" bw 0 : !hl.int< unsigned >
    // CHECK: hl.field @b2 bw 6 : !hl.int< unsigned >
    // CHECK: hl.field @b3 bw 15 : !hl.int< unsigned >
    unsigned b1 : 5;
    unsigned    : 0; // start a new unsigned int
    unsigned b2 : 6;
    unsigned b3 : 15;
};
