// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %file-check %s

// CHECK: hl.func @abort
// CHECK: hl.func @abort
// CHECK: hl.func @main

void abort();
void abort(){}
int main() {}
