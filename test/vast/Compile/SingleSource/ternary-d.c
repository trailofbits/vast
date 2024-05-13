// RUN: %vast-front -o %t %s
// RUN: %t 1 2 | %file-check %s -check-prefix="FOO"
// RUN: %t 1 2 3 | %file-check %s -check-prefix="BOO"
// RUN: %t 1 | %file-check %s -check-prefix="GOO"

#include <stdio.h>

int foo() { printf("foo\n"); return 0; }
int boo() { printf("boo\n"); return 1; }
int goo() { printf("goo\n"); return 2; }

int main(int argc, char **argv)
{
    // FOO: foo
    // BOO: boo
    // GOO: goo
    int o = (argc % 2 == 0) ? ( (argc == 2) ? goo()
                                            : boo() )
                            : foo();
    if (o == 0)
        return (argc == 3) ? 0 : 1;

    if (o == 1)
        return (argc == 4) ? 0 : 1;

    if (o == 2)
        return (argc == 2) ? 0 : 1;
}
