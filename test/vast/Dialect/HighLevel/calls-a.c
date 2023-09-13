// RUN: %vast-cc --ccopts -xc --from-source %s | %file-check %s
// RUN: %vast-cc --ccopts -xc --from-source %s > %t && %vast-opt %t | diff -B %t -

int constant() { return 7; }


void noop() {}


int add(int a, int b) { return a + b; }


int forward_decl(int a);


int main()
{
    // CHECK: hl.call @constant() : () -> !hl.int
    int c = constant();

    // CHECK: hl.call @noop() : () -> !hl.void
    noop();

    // CHECK: hl.call @add([[V1:%[0-9]+]], [[V2:%[0-9]+]]) : (!hl.int, !hl.int) -> !hl.int
    int v = add(1, 2);

    // CHECK: hl.call @forward_decl([[V3:%[0-9]+]])
    forward_decl(7);
}

int forward_decl(int a) { return a; }
