// RUN: %vast-cc --ccopts -xc --from-source %s | %vast-opt --vast-hl-lower-types --vast-hl-structs-to-llvm | %file-check %s

// CHECK: hl.typedef "X" : !llvm.struct<"X", (i32, ptr<struct<"X">>)>
struct X { int a; struct X *x; };
