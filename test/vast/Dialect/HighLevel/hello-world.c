// RUN: %vast-front %s -vast-emit-mlir=hl -o - | %file-check %s

// CHECK: hl.func @printf
#include <stdio.h>

// CHECK: hl.func @main {{.*}} () -> !hl.int
int main() {
    // CHECK: hl.call @printf
    printf("hello world\n");
    return 0;
}
