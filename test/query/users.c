// RUN: vast-cc --ccopts -xc --from-source %s > %t && \
// RUN: vast-query --symbol-users=a %t | \
// RUN: FileCheck %s -check-prefix=MAIN -check-prefix=FOO

// RUN: vast-cc --ccopts -xc --from-source %s > %t && \
// RUN: vast-query --symbol-users=a --scope=main %t | \
// RUN: FileCheck %s -check-prefix=MAIN

// RUN: vast-cc --ccopts -xc --from-source %s > %t && \
// RUN: vast-query --symbol-users=a --scope=foo %t | \
// RUN: FileCheck %s -check-prefix=FOO

// FOO: hl.decl.ref %0 : !hl.lvalue<!hl.int>
int foo() {
    int a;
    return a;
}

// MAIN: hl.decl.ref %0 : !hl.lvalue<!hl.int>
// MAIN: hl.decl.ref %0 : !hl.lvalue<!hl.int>
int main()
{
    int a = 1, b = 1;
    {
        int c = a + b;
    }

    int d = a + 7;
}
