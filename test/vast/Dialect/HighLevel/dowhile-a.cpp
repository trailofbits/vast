// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

// CHECK-LABEL: hl.func @_Z5basicv
void basic() {
    // CHECK: hl.do {
    do {
    } while (true);
    // CHECK: } while {
    // CHECK: [[V1:%[0-9]+]] = hl.const #true
    // CHECK: hl.cond.yield [[V1]]
    // CHECK: }
}

// CHECK-LABEL: hl.func @_Z10inner_condv
void inner_cond() {
    // CHECK: hl.var @i : !hl.lvalue<!hl.int>
    int i = 0;
    // CHECK: hl.do {
    // CHECK: [[V1:%[0-9]+]] = hl.ref @i
    // CHECK: hl.post.inc [[V1]]
    do {
        i++;
    } while (i < 100);
    // CHECK: } while {
    // CHECK: [[V2:%[0-9]+]] = hl.cmp slt
    // CHECK: hl.cond.yield [[V2]]
    // CHECK: }
}
