// RUN: %vast-front -o %t %s
// RUN: %t 1 2 | %file-check %s -check-prefix="FOO"
// RUN: %t 1 2 3 | %file-check %s -check-prefix="BOO"
// RUN: %t 1 | %file-check %s -check-prefix="GOO"

#include <stdio.h>

void foo() { printf("foo\n"); }
void boo() { printf("boo\n"); }
void goo() { printf("goo\n"); }

int main(int argc, char **argv)
{
    // FOO: foo
    // BOO: boo
    // GOO: goo
    (argc % 2 == 0) ? ( (argc == 2) ? goo()
                                    : boo() )
                    : foo();
}
