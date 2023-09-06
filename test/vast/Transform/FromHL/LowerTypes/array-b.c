// RUN: %vast-cc --ccopts -xc --from-source %s | %vast-opt --vast-hl-lower-types | FileCheck %s

// CHECK: hl.var "a" sc_extern : !hl.lvalue<memref<?xsi32>>
extern int a[];
