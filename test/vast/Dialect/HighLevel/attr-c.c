// RUN: %vast-front -vast-emit-mlir=hl -o - %s | %file-check %s
// RUN: %vast-front -vast-emit-mlir=hl -o - %s > %t && %vast-opt %t | diff -B %t -

// CHECK: hl.func @malloc {{.*}} attributes {alloc_size = #hl.alloc_size<size_pos : 1>, builtin = #hl.builtin<836>, malloc = #hl.restrict, nothrow = #hl.nothrow, sym_visibility = "private"}
#include <stdlib.h>

int main() {
    int *x = malloc(sizeof(*x));
    return 0;
}
