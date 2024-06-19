// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

void atomic_cast() {
    _Atomic int ai;
    _Atomic int bi;
    // CHECK: AtomicToNonAtomic : !hl.atomic<!hl.int> -> !hl.int
    // CHECK: NonAtomicToAtomic : !hl.int -> !hl.atomic<!hl.int>
    ai = bi;
    // CHECK: AtomicToNonAtomic : !hl.atomic<!hl.int> -> !hl.int
    int i = ai;
    // CHECK: NonAtomicToAtomic : !hl.int -> !hl.atomic<!hl.int>
    ai = i;
    // CHECK: AtomicToNonAtomic : !hl.atomic<!hl.int> -> !hl.int
    // CHECK: IntegralCast : !hl.int -> !hl.long
    long l = ai;
    // CHECK: IntegralCast : !hl.long -> !hl.int
    // CHECK: NonAtomicToAtomic : !hl.int -> !hl.atomic<!hl.int>
    ai = l;
}