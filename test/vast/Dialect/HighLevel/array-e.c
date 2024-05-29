// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

void foo(int size) {
    // CHECK: hl.var @arr : !hl.lvalue<!hl.array<?, !hl.int>> allocation_size {
    // CHECK:   hl.ref %arg0
    int arr[size];
}

extern int size();

void bar() {
    // CHECK: hl.var @arr : !hl.lvalue<!hl.array<?, !hl.int>> allocation_size {
    // CHECK:   hl.call @size() : () -> !hl.int
    int arr[4 + size()];
}

void baz(int x) {
    // CHECK: hl.var @Y : !hl.lvalue<!hl.array<?, !hl.int>> allocation_size
    // CHECK:   hl.ref %arg0
    int Y[x];
    ++x;
    // CHECK: hl.var @Z : !hl.lvalue<!hl.array<?, !hl.int>> allocation_size
    // CHECK:   hl.ref %arg0
    int Z[x];
}
