// RUN: %vast-front -vast-emit-mlir=hl -o - %s | %file-check %s

#include <stdlib.h>

int main() {
    // CHECK: hl.var @x : !hl.lvalue<!hl.ptr<!hl.int>>
    // CHECK: hl.sizeof.expr
    // CHECK: hl.ref @x
    int *x = malloc(sizeof(*x));
    return 0;
}
