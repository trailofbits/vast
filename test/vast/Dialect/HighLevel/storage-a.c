// RUN: vast-cc --ccopts -xc --from-source %s | FileCheck %s
// RUN: vast-cc --ccopts -xc --from-source %s > %t && vast-opt %t | diff -B %t -

// CHECK: hl.var @i : !hl.int
int i;

// CHECK: hl.var @ei sc_extern : !hl.int
extern int ei;

// CHECK: hl.var @si sc_static : !hl.int
static int si = 0;

// CHECK: hl.var @ti tsc_c_thread : !hl.int
_Thread_local int ti;

// CHECK: hl.var @tei sc_extern tsc_c_thread
_Thread_local extern int tei;

// CHECK: hl.var @sti sc_static tsc_c_thread
_Thread_local static int sti = 0;

void foo() {
    // CHECK: hl.var @si sc_static : !hl.int
    static int si = 0;

    // CHECK: hl.var @ri sc_register : !hl.int
    register int ri;

    // CHECK: hl.var @ai sc_auto : !hl.int
    auto int ai;

    // CHEcK: hl.var @tsi sc_static tsc_c_thread : !hl.int
    _Thread_local static int tsi;
}
