// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %vast-opt --vast-hl-lower-types | %file-check %s

// CHECK: hl.var "a" sc_extern : !hl.lvalue<memref<?xsi32>>
extern int a[];
