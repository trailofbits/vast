// RUN: %vast-front -vast-emit-mlir=hl -o - %s | %file-check %s
// REQUIRES: non-graph-regions

#include <stdlib.h>

int main() {
    // CHECK: hl.var @y : !hl.lvalue<!hl.ptr<!hl.int>>
    // CHECK: hl.sizeof.expr
    // CHECK: hl.ref [[Y]]
    int *y = malloc(sizeof(*y));
    if(1) {
    // CHECK: hl.var @x : !hl.lvalue<!hl.ptr<!hl.int>>
    // CHECK: hl.sizeof.expr
    // CHECK: hl.ref [[X]]
        int *x = malloc(sizeof(*x));
    }
    while(1) {
    // CHECK: hl.var @x : !hl.lvalue<!hl.ptr<!hl.int>>
    // CHECK: hl.sizeof.expr
    // CHECK: hl.ref [[X]]
        int *x = malloc(sizeof(*x));
    }
    do{
    // CHECK: hl.var @x : !hl.lvalue<!hl.ptr<!hl.int>>
    // CHECK: hl.sizeof.expr
    // CHECK: hl.ref [[X]]
        int *x = malloc(sizeof(*x));
    }while(0);
    // CHECK: core.scope
    // CHECK-NEXT: hl.var @x : !hl.lvalue<!hl.ptr<!hl.int>>
    // CHECK: hl.sizeof.expr
    // CHECK: hl.ref [[X]]
    for(int *x = malloc(sizeof(*x)); *x<100; *x++) {
    // CHECK: do
    // CHECK-NEXT: hl.var @f : !hl.lvalue<!hl.ptr<!hl.int>>
    // CHECK: hl.sizeof.expr
    // CHECK: hl.ref [[F]]
        int *f = malloc(sizeof(*f));

    }
    switch (*y)
    default:
    if (*y) {
        case 2: case 3: case 5: case 7: {
    // CHECK: hl.case
    // CHECK: hl.var @x : !hl.lvalue<!hl.ptr<!hl.int>>
    // CHECK: hl.sizeof.expr
    // CHECK: hl.ref [[X]]
            int *x = malloc(sizeof(*x));
            }
    }
    else {
        case 4: case 6: case 8: case 9: case 10:
            *y++;
    }
    switch(*y) {
        case 1:
        default: {
    // CHECK: hl.default
    // CHECK: hl.var @z : !hl.lvalue<!hl.ptr<!hl.int>>
    // CHECK: hl.sizeof.expr
    // CHECK: hl.ref [[Z]]
            int *z = malloc(sizeof(*z));
        }
    }
    // CHECK: hl.var @g : !hl.lvalue<!hl.ptr<!hl.int>>
    // CHECK: hl.cond
    // CHECK: ?
    // CHECK: hl.sizeof.expr
    // CHECK: hl.ref [[G]]
    // CHECK: hl.ref [[Y]]

    int *g = y ? malloc(sizeof(*g)) : y ;
    ({
    // CHECK: hl.stmt.expr
    // CHECK: hl.var @z : !hl.lvalue<!hl.ptr<!hl.int>>
    // CHECK: hl.sizeof.expr
    // CHECK: hl.ref [[Z]]
            int *z = malloc(sizeof(*z));
     });
    return 0;
}
