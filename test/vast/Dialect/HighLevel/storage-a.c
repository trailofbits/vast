// RUN: vast-cc --ccopts -xc --from-source %s | FileCheck %s
// RUN: vast-cc --ccopts -xc --from-source %s > %t && vast-opt %t | diff -B %t -

// CHECK: hl.var "i" : !hl.lvalue<!hl.int>
int i;

// CHECKL: hl.var "ei extern" : !hl.lvalue<!hl.int>
extern int ei;

// CHECK: hl.var "si static" : !hl.lvalue<!hl.int>
static int si = 0;

// CHECK: hl.var "ti thread_local" : !hl.lvalue<!hl.int>
_Thread_local int ti;

// CHECK: hl.var @tei extern thread_local
_Thread_local extern int tei;

// CHECK: hl.var @sti static thread_local
_Thread_local static int sti = 0;

void foo() {
    // CHECK: hl.var "si static" : !hl.lvalue<!hl.int>
    static int si = 0;

    // CHECK: hl.var "ri register" : !hl.lvalue<!hl.int>
    register int ri;

    // CHECK: hl.var "ai auto" : !hl.lvalue<!hl.int>
    auto int ai;

    // CHEcK: hl.var "tsi static thread_local" : !hl.lvalue<!hl.int>
    _Thread_local static int tsi;
}
