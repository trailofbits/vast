// RUN: %vast-front -vast-emit-mlir=hl -o - %s | %file-check %s

// CHECK: hl.func
// CHECK: core.scope {
// CHECK-NEXT: }
// CHECK: core.implicit.return
void startrek_user_init(void){{}}
