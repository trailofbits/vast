// RUN: %vast-front -o %t %s
// RUN: %t 1 2 | %file-check %s -check-prefix="A"
// RUN: %t 1 2 3 | %file-check %s -check-prefix="B"
// RUN: %t | %file-check %s -check-prefix="A"

#include <stdio.h>

int main(int argc, char **argv)
{
    char c = (argc % 2 == 0) ? '0' : '1';
    // A: 1
    // B: 0
    putchar(c);
}
