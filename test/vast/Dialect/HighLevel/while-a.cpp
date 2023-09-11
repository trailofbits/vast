// RUN: %vast-cc --from-source %s | FileCheck %s
// RUN: %vast-cc --from-source %s > %t && %vast-opt %t | diff -B %t -

// CHECK-LABEL: hl.func @_Z11while_emptyv
void while_empty()
{
    // CHECK: hl.while
    // CHECK: [[V1:%[0-9]+]] = hl.const #hl.bool<true> : !hl.bool
    // CHECK: hl.cond.yield [[V1]]
    while (true) {
        // CHECK: } do {
    }
}

// CHECK-LABEL: hl.func @_Z12while_simplei
void while_simple(int a)
{
    // CHECK: hl.while
    // CHECK: [[V1:%[0-9]+]] = hl.cmp sgt
    // CHECK: hl.cond.yield [[V1]]
    while (a > 0)
    {
        // CHECK: } do {
        a = a - 1;
        // CHECK: [[V2:%[0-9]+]] = hl.sub
        // CHECK: hl.assign [[V2]]
    }
}

// CHECK-LABEL: hl.func @_Z12while_nestedii
void while_nested(int a, int b)
{
    // CHECK: hl.while
    // CHECK: [[V1:%[0-9]+]] = hl.cmp sgt
    // CHECK: hl.cond.yield [[V1]]
    while (a > 0) {
        // CHECK: } do {
        int c = b;
        // CHECK: hl.while
        // CHECK: [[V2:%[0-9]+]] = hl.cmp sgt
        // CHECK: hl.cond.yield [[V2]]
        while (c > 0) {
            // CHECK: } do {
            c = c - 1;
        }
        a = a - 1;
    }
}
