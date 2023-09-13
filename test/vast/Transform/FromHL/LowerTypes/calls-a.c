// RUN: %vast-cc --ccopts -xc --from-source %s | %vast-opt --vast-hl-lower-types | %file-check %s

int constant() { return 7; }


void noop() {}


int add(int a, int b) { return a + b; }


int forward_decl(int a);


int main()
{
    // CHECK: hl.call @constant() : () -> si32
    int c = constant();

    // CHECK: hl.call @noop() : ()
    noop();

    // CHECK: hl.call @add([[V1:%[0-9]+]], [[V2:%[0-9]+]]) : (si32, si32) -> si32
    int v = add(1, 2);

    // CHECK: hl.call @forward_decl([[V3:%[0-9]+]]) : (si32) -> si32
    forward_decl(7);
}

int forward_decl(int a) { return a; }
