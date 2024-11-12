// RUN: %vast-front %s -vast-emit-mlir=hl -o - | %file-check %s
// RUN: %vast-front %s -vast-emit-mlir=hl -o - > %t && %vast-opt %t | diff -B %t -

int load(int* p) {
    // CHECK: hl.atomic_expr "__atomic_load_n" : !hl.int
    // CHECK: hl.ref @p
    // CHECK: hl.const #core.integer<5>
    int q = __atomic_load_n (p, __ATOMIC_SEQ_CST);
    return q;
}
