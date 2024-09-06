// RUN: %vast-front %s -vast-emit-mlir=hl -o - | %file-check %s
// RUN: %vast-front %s -vast-emit-mlir=hl -o - > %t && %vast-opt %t | diff -B %t -

// CHECK: hl.alias @Int : !hl.int
using Int = int;
Int variable = 0;
