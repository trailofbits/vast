// RUN: %vast-front -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-front -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

int main() {
    void foo();
}

// CHECK: hl.func @foo external ([[ARG0:%[a-z0-9]+]]: !hl.lvalue<!hl.int>)
void foo(int n){}
