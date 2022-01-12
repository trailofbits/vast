// RUN: vast-cc --from-source %s | FileCheck %s
// RUN: vast-cc --from-source %s > %t && vast-opt --vast-hl-lower-types %t | diff -B %t -

int constant() { return 7; }


void noop() {}


int add(int a, int b) { return a + b; }


int forward_decl(int a);


int main()
{
    // CHECK: hl.call @constant() : () -> i32
    int c = constant();

    // CHECK: hl.call @noop() : () -> none
    noop();

    // CHECK: hl.call @add([[V1:%[0-9]+]], [[V2:%[0-9]+]]) : (i32, i32) -> i32
    int v = add(1, 2);

    // CHECK: hl.call @forward_decl([[V3:%[0-9]+]]) : (i32) -> i32
    forward_decl(7);
}

int forward_decl(int a) { return a; }
