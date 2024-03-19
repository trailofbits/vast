// RUN: %vast-front -o %t %s && %t hello | %file-check %s

#include <stdio.h>

int main(int argc, char **argv)
{
    // CHECK: argc: 2
    printf("argc: %i\n", argc);
}
