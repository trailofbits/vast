// RUN: %vast-front -vast-emit-mlir=hl %s -o - | %vast-opt --vast-hl-ude | %file-check %s

// CHECK-NOT: hl.func @unused_declaration
void unused_declaration(void);

// CHECK: hl.func @unused_definition
void unused_definition(void) {}