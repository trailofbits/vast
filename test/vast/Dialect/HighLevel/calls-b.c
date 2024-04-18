// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

// CHECK: @factorial {{.*}} ([[A1:%arg[0-9]+]]: !hl.lvalue<!hl.int>) -> !hl.int
int factorial(int i) {
    // CHECK: hl.if
    if ( i <= 1 )
        return 1;
    // CHECK: hl.call @factorial([[V1:%[0-9]+]]) : (!hl.int) -> !hl.int
    return i * factorial(i - 1);
}

// CHECK-NOT: @factorial
