// RUN: vast-cc --ccopts -xc --from-source %s | FileCheck %s
// RUN: vast-cc --ccopts -xc --from-source %s > %t && vast-opt %t | diff -B %t -

int ptr_cmp(void *a, void *b) {
    // CHECK: hl.cmp eq !hl.ptr<!hl.void> [[A:%[0-9]+]], [[B:%[0-9]+]] -> !hl.int
    return a == b;
}
