// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %vast-opt --vast-hl-lower-types --vast-hl-structs-to-llvm | %file-check %s

// CHECK: hl.type "Y"
struct Y;
// CHECK: hl.typedef "X" : !llvm.struct<"X", (i32, ptr<struct<"Y", opaque>>)>
struct X { int x; struct Y *y; };
