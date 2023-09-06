// RUN: %vast-cc --ccopts -xc --from-source %s | %vast-opt --vast-hl-dce --vast-hl-lower-types --vast-hl-resolve-typedefs | FileCheck %s

typedef int INT;

// CHECK: hl.func external @foo (!hl.lvalue<si32>) -> si32 attributes {sym_visibility = "private"}
INT foo(INT);
