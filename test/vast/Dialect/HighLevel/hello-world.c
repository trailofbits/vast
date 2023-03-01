// RUN: vast-front %s -vast-emit-high-level -o - | FileCheck %s

// CHECK: hl.func external @printf
#include <stdio.h>

// CHECK: hl.func external @main () -> !hl.int
int main() {
    // CHECK: hl.call @printf
    printf("hello world\n");
    return 0;
}
