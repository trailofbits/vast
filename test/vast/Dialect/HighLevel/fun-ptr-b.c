// RUN: %vast-front -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-front -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

int hash_proof_destroy_noop(int x) {return x;}

int main() {
    int x = 0;
    // CHECK: hl.cond : !hl.ptr<!core.fn<(!hl.lvalue<!hl.int>) -> (!hl.int)>>
    void *fn = x ? ((void *)0) : hash_proof_destroy_noop;
}

