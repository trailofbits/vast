// RUN: vast-front -vast-emit-mlir=hl -o - %s

#include <stdlib.h>

int main() {
    int *y = malloc(sizeof(*y));
    if(1) {
        int *x = malloc(sizeof(*x));
    }
    while( 1) {
        int *x = malloc(sizeof(*x));
    }
    do{
        int *x = malloc(sizeof(*x));
    }while(0);
    for(int *x = malloc(sizeof(*x)); *x<100; *x++) {
        int *f = malloc(sizeof(*f));

    }
    switch (*y)
    default:
    if (*y) {
        case 2: case 3: case 5: case 7:
            *y++;
            int *x = malloc(sizeof(*x));
    }
    else {
        case 4: case 6: case 8: case 9: case 10:
            *y++;
    }
    switch(*y)
        case 1:
            *y++;
            int *x = malloc(sizeof(*x));
    switch(*y) {
        case 1:
            *y++;
            int *x = malloc(sizeof(*x));
        case 2:
        default:
            *y++;
            int *z = malloc(sizeof(*z));
    }
    int *g = y ? malloc(sizeof(*g)): y ;
    ({
            int *z = malloc(sizeof(*z));
     });
    return 0;
}
