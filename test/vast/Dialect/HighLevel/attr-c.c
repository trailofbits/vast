// RUN: %vast-front -vast-emit-mlir=hl -o - %s | %file-check %s
// RUN: %vast-front -vast-emit-mlir=hl -o - %s > %t && %vast-opt %t | diff -B %t -

// CHECK: hl.func @malloc {{.*}} 
#include <stdlib.h>

int main() {
    int *x = malloc(sizeof(*x));
    return 0;
}
