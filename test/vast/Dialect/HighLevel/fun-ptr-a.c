// RUN: %vast-front -vast-emit-mlir=hl -o - %s | %file-check %s

void f(void g(void)) {
    // CHECK: hl.indirect_call {{.*}} : !hl.decayed<!hl.ptr<!core.fn<() -> (!hl.void)>>>() : () -> !hl.void
    g();
}

void h(void) {}

int main (void) {
    // CHECK: FunctionToPointerDecay
    // CHECK: hl.call {{.*}} : (!hl.ptr<!core.fn<() -> (!hl.void)>>) -> !hl.void
    f(h);
    return 0;
}
