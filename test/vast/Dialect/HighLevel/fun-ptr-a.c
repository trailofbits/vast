// RUN: vast-front -vast-emit-mlir=hl -o - %s | FileCheck %s

void f(void g(void)) {
    // CHECK: {{.*hl.indirect_call .* : !hl.decayed<!hl.ptr<\(\) -> !hl.void>>\(\) : \(\) -> !hl.void}}
    g();
}

void h(void) {}

int main (void) {
    // CHECK: {{.*hl.implicit_cast .* FunctionToPointerDecay.*}}
    // CHECK: {{.*hl.call .* : \(!hl.lvalue<!hl.ptr<\(\) -> !hl.void>>\) -> !hl.void}}
    f(h);
    return 0;
}
