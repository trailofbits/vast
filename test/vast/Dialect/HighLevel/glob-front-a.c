// RUN: %vast-front %s -vast-emit-high-level -o - | %file-check %s

// CHECK: hl.var "NUM"
// CHECK: hl.value.yield
short NUM = 10;
//CHECK: hl.globref "NUM"
//CHECK: hl.assign
int main() {NUM = 11;}
