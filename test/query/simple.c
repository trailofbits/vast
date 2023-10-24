// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o %t && %vast-query --show-symbols=functions %t | %file-check %s

// CHECK: func : main
int main() {}
