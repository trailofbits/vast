// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

// CHECK: hl.func @foo external
void foo();
// CHECK: hl.call @bar


// CHECK: hl.func @bar external () -> !hl.void attributes {sym_visibility = "private"}
void bar();

void foo() {
    bar();
}

void foo();