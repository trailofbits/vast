// RUN: %vast-front -vast-emit-mlir=hl -o - %s | %file-check %s

#include <stdlib.h>

int main() {
    // CHECK: [[X:%[0-9]+]] = hl.var @x : !hl.lvalue<!hl.ptr<!hl.int>>
    // CHECK: hl.sizeof.expr
    // CHECK: hl.ref [[X]]
    int *x = malloc(sizeof(*x));
    return 0;
}
