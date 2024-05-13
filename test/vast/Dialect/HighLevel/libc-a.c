// RUN: %vast-front -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-front -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

#include <stdio.h>

int main() {
    // CHECK: hl.call @printf
    printf("hello world\n");
}
