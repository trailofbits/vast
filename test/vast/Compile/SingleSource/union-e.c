// RUN: %vast-front -o %t %s && %t | %file-check %s

#include <stdio.h>

union U
{
    int arr[2];
};

int main(int argc, char **argv)
{
    union U u;
    u.arr[0] = 0;
    u.arr[1] = 0;

    // CHECK: 0 0
    printf("%i %i\n", u.arr[0], u.arr[1]);

    u.arr[0] = 10;
    u.arr[1] = 100;

    // CHECK: 10 100
    printf("%i %i\n", u.arr[0], u.arr[1]);

    u.arr[1] = -100;

    // CHECK: 10 -100
    printf("%i %i\n", u.arr[0], u.arr[1]);

    union U u1;
    u1.arr[0] = 42;
    u1.arr[1] = 43;

    u = u1;

    // CHECK: 42 43
    printf("%i %i\n", u.arr[0], u.arr[1]);
}
