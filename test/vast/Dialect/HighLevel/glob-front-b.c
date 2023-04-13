// RUN: vast-front %s -vast-emit-high-level -o - | FileCheck %s

// CHECK: hl.var "NUM"
short NUM;
//CHECK: hl.globref "NUM"
//CHECK: hl.assign
int main() {NUM = 10;}
