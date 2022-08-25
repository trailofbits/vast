// RUN: vast-cc --ccopts -xc --from-source %s | vast-opt --vast-hl-lower-types --vast-hl-structs-to-llvm | FileCheck %s

// CHECK: hl.type.decl "Y"
struct Y;
// CHECK: hl.typedef "X" : !llvm.struct<"X", (i32, ptr<struct<"Y", opaque>>)>
struct X { int x; struct Y *y; };
