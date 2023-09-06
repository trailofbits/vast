// RUN: %vast-front -o %t %s && (%t 1; test $? -eq 2)
// REQUIRES: union_lowering

#include <stdlib.h>

int putchar(int);
int main(int argc, char **argv) {
    int x = 0;
    if (argc > 1) {
        int *y = malloc(sizeof(*y));
    }
    for (int i = 0; i < argc; i++){
        int *p = malloc(sizeof(*p));
        putchar('a');
        putchar('\n');
    }
    return argc;
}
