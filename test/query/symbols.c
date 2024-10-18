// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o %t && \
// RUN: %vast-query --show-symbols=all %t | \
// RUN: %file-check %s -check-prefix=FOO-VAR -check-prefix=MAIN-VAR -check-prefix=FUN

// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o %t && \
// RUN: %vast-query --show-symbols=vars %t | \
// RUN: %file-check %s -check-prefix=FOO-VAR -check-prefix=MAIN-VAR

// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o %t && \
// RUN: %vast-query --show-symbols=vars %t --scope=foo | \
// RUN: %file-check %s -check-prefix=FOO-VAR

// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o %t && \
// RUN: %vast-query --show-symbols=vars %t --scope=main | \
// RUN: %file-check %s -check-prefix=MAIN-VAR

// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o %t && \
// RUN: %vast-query --show-symbols=functions %t | \
// RUN: %file-check %s -check-prefix=FUN

// FOO-VAR-DAG: hl.var : a
// FUN-DAG: func : foo
int foo() {
    int a;
    return a;
}

// MAIN-VAR-DAG: hl.var : a
// MAIN-VAR-DAG: hl.var : b
// MAIN-VAR-DAG: hl.var : c
// FUN-DAG: func : main
int main()
{
    int a = 1, b = 1;
    {
        int c = a + b;
    }
}
