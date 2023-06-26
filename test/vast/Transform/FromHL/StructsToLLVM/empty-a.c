// RUN: vast-cc --ccopts -xc --from-source %s | vast-opt --vast-hl-splice-trailing-scopes --vast-hl-lower-types --vast-hl-structs-to-llvm | FileCheck %s

// CHECK: hl.typedef "X" : !llvm.struct<"X", ()>
struct X {};
