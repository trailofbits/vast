// RUN: %vast-front -vast-emit-mlir=hl -fcf-protection %s -o - | %file-check %s
// RUN: %vast-front -vast-emit-mlir=hl -fcf-protection %s -o %t && %vast-opt %t | diff -B %t -
// REQUIRES: target-x86_64

// CHECK: hl.nocf_check = #hl.nocf_check
__attribute__((nocf_check))
void fn2(void) {}
