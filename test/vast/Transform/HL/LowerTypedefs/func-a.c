// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %vast-opt --vast-hl-dce --vast-hl-lower-types --vast-hl-lower-elaborated-types --vast-hl-lower-typedefs | %file-check %s

typedef int INT;

// CHECK: hl.func @foo (!hl.lvalue<si32>) -> si32 attributes {sym_visibility = "private"}
INT foo(INT);
