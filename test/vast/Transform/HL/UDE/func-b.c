// RUN: %vast-front -vast-emit-mlir=hl %s -o - | %vast-opt --vast-hl-ude | %file-check %s

// CHECK: hl.func @used_declaration
void used_declaration(void);

void use(void) { used_declaration(); }