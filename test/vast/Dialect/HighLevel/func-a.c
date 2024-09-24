// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

// CHECK: hl.func @foo
void foo();


// CHECK: hl.func @bar
void bar();

// CHECK: hl.func @foo
void foo() {
    // CHECK: hl.call @bar
    bar();
}
