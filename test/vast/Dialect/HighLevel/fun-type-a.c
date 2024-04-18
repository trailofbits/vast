// RUN: %vast-front -x c -vast-emit-mlir=hl -o - %s | %file-check %s

// CHECK: hl.func @fun {{.*}} (%arg0: !hl.lvalue<!hl.int>, %arg1: !hl.lvalue<!hl.int>) -> !hl.ptr<!hl.int> {
int *(fun)(int a, int b){int c = a + b; return &c;}
int main() {return 0;}
