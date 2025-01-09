// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o %t && \
// RUN: %vast-query --symbol-users=a %t | \
// RUN: %file-check %s -check-prefix=MAIN -check-prefix=FOO

// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o %t && \
// RUN: %vast-query --symbol-users=a --scope=main %t | \
// RUN: %file-check %s -check-prefix=MAIN

// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o %t && \
// RUN: %vast-query --symbol-users=a --scope=foo %t | \
// RUN: %file-check %s -check-prefix=FOO


// FOO: hl.ref @a
int foo() {
    int a;
    return a;
}

// MAIN: hl.ref @a
int main()
{
    int a = 1, b = 1;
    {
        int c = a + b;
    }

    int d = a + 7;
}
