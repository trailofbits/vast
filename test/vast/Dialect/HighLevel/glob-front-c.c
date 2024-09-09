// RUN: %vast-front %s -vast-emit-mlir=hl -o - | %file-check %s

extern short GIB_SHORT(void);
// CHECK: hl.var @NUM
short NUM;
//CHECK: hl.globref @NUM
//CHECK: hl.assign
int main() {NUM = GIB_SHORT();}
