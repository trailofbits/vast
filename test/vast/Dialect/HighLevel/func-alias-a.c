// RUN: %vast-cc1 -triple x86_64-unknown-linux-gnu -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-cc1 -triple x86_64-unknown-linux-gnu -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

// CHECK: hl.func @bar external
void bar(void) {}

// CHECK: hl.func @foo external {{.*}} attributes {hl.alias = #hl.alias<"bar">}
void foo(void) __attribute__((__alias__("bar")));
