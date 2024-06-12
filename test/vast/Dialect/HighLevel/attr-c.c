// RUN: %vast-front -vast-emit-mlir=hl -o - %s | %file-check %s
// RUN: %vast-front -vast-emit-mlir=hl -o - %s > %t && %vast-opt %t | diff -B %t -

#include <stdlib.h>
// CHECK: hl.func @my_malloc {{.*}}hl.alloc_size = #hl.alloc_size<size_pos : 2>
void *my_malloc(unsigned long x, unsigned long y) __attribute__((alloc_size(1)));
void *my_malloc(unsigned long x, unsigned long y) __attribute__((alloc_size(2)));
void *my_malloc(unsigned long x, unsigned long y) {return malloc(x);}

int main() {
    int *x = my_malloc(sizeof(*x), 6);
    return 0;
}
