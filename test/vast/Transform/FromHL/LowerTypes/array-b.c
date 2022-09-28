// RUN: vast-cc --ccopts -xc --from-source %s | vast-opt --vast-hl-lower-types | FileCheck %s
// REQUIRES: type-lowering

// CHECK: hl.var "a" sc_extern : !hl.lvalue<memref<?xi32>>
extern int a[];
