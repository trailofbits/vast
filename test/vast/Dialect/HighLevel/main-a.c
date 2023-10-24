// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

// CHECK: module @"{{.*}}vast/Dialect/HighLevel/main-a.c"
// CHECK: hl.func @main () -> !hl.int
int main() {}
