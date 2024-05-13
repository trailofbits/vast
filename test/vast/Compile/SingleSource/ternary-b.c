// RUN: %vast-front -o %t %s
// RUN: %t 1 2 | %file-check %s -check-prefix="FOO"
// RUN: %t 1 2 3 | %file-check %s -check-prefix="BOO"
// RUN: %t | %file-check %s -check-prefix="FOO"

#include <stdio.h>

void foo() { printf("foo\n"); }
void boo() { printf("boo\n"); }

int main(int argc, char **argv)
{
    // FOO: foo
    // BOO: boo
    (argc % 2 == 0) ? boo() : foo();
}
