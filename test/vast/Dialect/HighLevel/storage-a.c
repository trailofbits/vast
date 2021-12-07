// RUN: vast-cc --ccopts -xc --from-source %s | FileCheck %s
// RUN: vast-cc --ccopts -xc --from-source %s > %t && vast-opt %t | diff -B %t -

// CHECK: hl.global @i : !hl.int
int i;

// CHECKL: hl.global @ei extern : !hl.int
extern int ei;

// CHECK: hl.global @si static : !hl.int
static int si = 0;

// CHECK: hl.global @ti thread_local : !hl.int
_Thread_local int ti;

// CHECK: hl.global @tei extern thread_local
_Thread_local extern int tei;

// CHECK: hl.global @sti static thread_local
_Thread_local static int sti = 0;

void foo() {
    // CHECK: hl.var @si static : !hl.int
    static int si = 0;

    // CHECK: hl.var @ri register : !hl.int
    register int ri;

    // CHECK: hl.var @ai auto : !hl.int
    auto int ai;

    // CHEcK: hl.var @tsi static thread_local : !hl.int
    _Thread_local static int tsi;
}
