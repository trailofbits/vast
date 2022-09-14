// RUN: vast-cc --ccopts -xc --from-source %s | FileCheck %s
// RUN: vast-cc --ccopts -xc --from-source %s > %t && vast-opt %t | diff -B %t -

int ptr_cmp(void *a, void *b) {
    // CHECK: hl.cmp eq [[A:%[0-9]+]], [[B:%[0-9]+]] : !hl.ptr<!hl.void>, !hl.ptr<!hl.void> -> !hl.int
    return a == b;
}
