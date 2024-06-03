// RUN: %vast-front -vast-emit-mlir=hl -o - %s | %file-check %s
// RUN: %vast-front -vast-emit-mlir=hl -o - %s > %t && %vast-opt %t | diff -B %t -

// CHECK: hl.func @fun {{.*}} attributes {hl.warn_unused_result = #hl.warn_unused_result}
int __attribute__((warn_unused_result)) fun(void) {return 1;}
