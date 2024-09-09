// RUN: %vast-front %s -vast-emit-mlir=hl -o - | %file-check %s

// CHECK: hl.var @NUM
short NUM;
//CHECK: hl.globref @NUM
//CHECK: hl.assign
int main() {NUM = 10;}
