// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

// CHECK: hl.var @i, <external> : !hl.lvalue<!hl.int>
int i;

// CHECKL: hl.var @ei, <external> sc_extern : !hl.lvalue<!hl.int>
extern int ei;

// CHECK: hl.var @si, <internal> sc_static : !hl.lvalue<!hl.int>
static int si = 0;

// CHECK: hl.var @ti, <external> tsc_c_thread : !hl.lvalue<!hl.int>
_Thread_local int ti;

// CHECK: hl.var @tei, <external> sc_extern tsc_c_thread
_Thread_local extern int tei;

// CHECK: hl.var @sti, <internal> sc_static tsc_c_thread
_Thread_local static int sti = 0;

void foo() {
    // CHECK: hl.var @si sc_static : !hl.lvalue<!hl.int>
    static int si = 0;

    // CHECK: hl.var @ri sc_register : !hl.lvalue<!hl.int>
    register int ri;

    // CHECK: hl.var @ai sc_auto : !hl.lvalue<!hl.int>
    auto int ai;

    // CHEcK: hl.var @tsi sc_static tsc_c_thread : !hl.lvalue<!hl.int>
    _Thread_local static int tsi;
}
