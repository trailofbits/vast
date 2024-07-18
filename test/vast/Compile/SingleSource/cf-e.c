// RUN: %vast-front -o %t %s && %t | %file-check %s

int foo(int input) {
    int out = 0;

    while (--input >= 0) {
        if (input == 4)
            continue;
        if (input == 2)
            break;

        int a = input;
        while (--a >= 0) {
            if (a % 4 == 0)
                continue;
            if (a == 1)
                break;
            ++out;
        }
    }

    return out;
}

#include <stdio.h>

// CHECK: 0
// CHECK: 0
// CHECK: 0
// CHECK: 0
// CHECK: 1
// CHECK: 1
// CHECK: 3
// CHECK: 6
// CHECK: 10
// CHECK: 15

int main(int argc, char **argv)
{
    for (int i = 0; i < 10; ++i) {
        printf("%i\n", foo(i));
    }
}
