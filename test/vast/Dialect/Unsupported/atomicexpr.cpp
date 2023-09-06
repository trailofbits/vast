// RUN: %vast-front %s -vast-emit-mlir=hl -o - | FileCheck %s
// RUN: %vast-front %s -vast-emit-mlir=hl -o - > %t && %vast-opt %t | diff -B %t -

int load(int* p) {
    // CHECK: unsup.stmt "AtomicExpr"
    // CHECK: hl.ref %arg0
    // CHECK: hl.const #hl.integer<5>
    int q = __atomic_load_n (p, __ATOMIC_SEQ_CST);
    return q;
}

