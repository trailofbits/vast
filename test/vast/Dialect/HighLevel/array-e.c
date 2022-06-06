// RUN: vast-cc --ccopts -xc --from-source %s | FileCheck %s
// RUN: vast-cc --ccopts -xc --from-source %s > %t && vast-opt %t | diff -B %t -

void foo(int size) {
    // CHECK: hl.var "arr" : !hl.lvalue<!hl.array<?, !hl.int>> allocation_size {
    // CHECK:   hl.decl.ref %arg0 : !hl.lvalue<!hl.int>
    int arr[size];
}

extern int size();

void bar() {
    // CHECK: hl.var "arr" : !hl.lvalue<!hl.array<?, !hl.int>> allocation_size {
    // CHECK:   hl.call @size() : () -> !hl.int
    int arr[4 + size()];
}

void baz(int x) {
    // CHECK: hl.var "Y" : !hl.lvalue<!hl.array<?, !hl.int>> allocation_size
    // CHECK:   hl.decl.ref %arg0 : !hl.lvalue<!hl.int>
    int Y[x];
    ++x;
    // CHECK: hl.var "Z" : !hl.lvalue<!hl.array<?, !hl.int>> allocation_size
    // CHECK:   hl.decl.ref %arg0 : !hl.lvalue<!hl.int>
    int Z[x];
}
