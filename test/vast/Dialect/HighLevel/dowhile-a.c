// RUN: vast-cc --ccopts -xc --from-source %s | FileCheck %s
// RUN: vast-cc --ccopts -xc --from-source %s > %t && vast-opt %t | diff -B %t -

// CHECK-LABEL: hl.func external @basic
void basic() {
    // CHECK: hl.do {
    do {
    } while (true);
    // CHECK: } while {
    // CHECK: [[V1:%[0-9]+]] = hl.const #hl.bool<true> : !hl.bool
    // CHECK: hl.cond.yield [[V1]]
    // CHECK: }
}

// CHECK-LABEL: hl.func external @inner_cond
void inner_cond() {
    // CHECK: [[I:%[0-9]+]] = hl.var "i" : !hl.lvalue<!hl.int>
    int i = 0;
    // CHECK: hl.do {
    // CHECK: [[V1:%[0-9]+]] = hl.ref [[I]]
    // CHECK: hl.post.inc [[V1]]
    do {
        i++;
    } while (i < 100);
    // CHECK: } while {
    // CHECK: [[V2:%[0-9]+]] = hl.cmp slt
    // CHECK: hl.cond.yield [[V2]]
    // CHECK: }
}
