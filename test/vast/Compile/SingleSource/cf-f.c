// RUN: %vast-front -o %t %s && %t | %file-check %s

int _while(int a) {
    int sum = 0;
    while(--a >= 0) {
        ++sum;
    }
    return sum;
}

int _do_while(int a) {
    int sum = 0;
    do {
        ++sum;
    } while (--a >= 0);
    return sum;
}

#include <stdio.h>

// CHECK: 0, 1
// CHECK: 2, 3
// CHECK: 4, 5
// CHECK: 6, 7
// CHECK: 8, 9

int main(int argc, char **argv)
{
    for (int i = 0; i < 10; i += 2) {
        printf("%i, %i\n", _while(i), _do_while(i));
    }
}
