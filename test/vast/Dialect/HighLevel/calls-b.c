// RUN: vast-cc --from-source %s | FileCheck %s
// RUN: vast-cc --from-source %s > %t && vast-opt %t | diff -B %t -

// CHECK: @factorial({{.*}}: !hl.int) -> !hl.int
int factorial(int i) {
    // CHECK: hl.if
    if ( i <= 1 )
        return 1;
    // CHECK: hl.call @factorial([[V1:%[0-9]+]]) : (!hl.int) -> !hl.int
    return i * factorial(i - 1);
}

// CHECK-NOT: @factorial
