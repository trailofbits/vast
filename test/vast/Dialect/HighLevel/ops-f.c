// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

int ptr_cmp(void *a, void *b) {
    // CHECK: hl.cmp eq [[A:%[0-9]+]], [[B:%[0-9]+]] : !hl.ptr<!hl.void>, !hl.ptr<!hl.void> -> !hl.int
    return a == b;
}
