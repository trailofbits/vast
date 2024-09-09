// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

short a;

void foo() {
    // CHECK: [[A:%[0-9]+]] = hl.globref @a : !hl.lvalue<!hl.short>
    // CHECK: [[C:%[0-9]+]] = hl.const #core.integer<4> : !hl.int
    // CHECK: hl.assign.add [[C]] to [[A]] : !hl.int, !hl.lvalue<!hl.short> -> !hl.short
    a += 4;
}
