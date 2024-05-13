// RUN: %vast-front -o %t %s && %t hello | %file-check %s

#include <stdio.h>

void boo() {
    printf("boo reached!");
}

void foo() {
    return boo();
}

int main(int argc, char **argv)
{
    // CHECK: boo reached!
    foo();
}
