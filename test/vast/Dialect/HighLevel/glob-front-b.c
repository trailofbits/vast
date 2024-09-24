// RUN: %vast-front %s -vast-emit-mlir=hl -o - | %file-check %s

// CHECK: hl.var @NUM
short NUM;
//CHECK: hl.ref @NUM
//CHECK: hl.assign
int main() {NUM = 10;}
