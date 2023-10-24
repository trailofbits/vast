// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %vast-opt --vast-hl-lower-types --vast-hl-structs-to-llvm | %file-check %s

// CHECK: hl.typedef "X" : !llvm.struct<"X", (i32, ptr<struct<"X">>)>
struct X { int a; struct X *x; };
