// RUN: %vast-front -vast-emit-mlir=hl -x c -o - %s | FileCheck %s
// RUN: %vast-front -vast-emit-mlir=hl -x c++ -o - %s | FileCheck %s

// CHECK: hl.func {{.*}} () -> !hl.void
void f1() {}
// CHECK: hl.func {{.*}} () -> !hl.void
void f2(void) {}
// CHECK: hl.func {{.*}} () -> !hl.int
int  f3(void) {return 1;}

