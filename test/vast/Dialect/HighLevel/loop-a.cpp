// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

// CHECK-LABEL: hl.func @_Z11loop_simplev
void loop_simple()
{
    // CHECK: core.scope {
        // CHECK: hl.var @i : !hl.lvalue<!hl.int>
        // CHECK: hl.for {
        // CHECK:   hl.cmp slt
        // CHECK: } incr {
        // CHECK:   hl.post.inc
        // CHECK: } do {
        // CHECK: }
    for (int i = 0; i < 100; i++) {}
    // CHECK: }

    int after_loop;
}

// CHECK-LABEL: hl.func @_Z11loop_noinitv
void loop_noinit()
{
    int i = 0;
    // CHECK: hl.var @i : !hl.lvalue<!hl.int>
    // CHECK: hl.for {
    // CHECK:   [[V2:%[0-9]+]] = hl.cmp slt
    // CHECK:   hl.cond.yield [[V2]]
    // CHECK: } incr {
    // CHECK:   hl.pre.inc
    // CHECK: } do {
    // CHECK: }
    for (;i < 100; ++i) {}
}

// CHECK-LABEL: hl.func @_Z11loop_noincrv
void loop_noincr()
{
    // CHECK: hl.var @i : !hl.lvalue<!hl.int>
    // CHECK: hl.for {
    // CHECK:   [[V2:%[0-9]+]] = hl.cmp slt
    // CHECK:   hl.cond.yield [[V2]]
    // CHECK: } incr {
    // CHECK: } do {
    // CHECK:   hl.pre.inc
    // CHECK: }
    for (int i = 0; i < 100;) { ++i; }
}

// CHECK-LABEL: hl.func @_Z13loop_infinitev
void loop_infinite()
{
    // CHECK: hl.for {
    // CHECK:   [[V1:%[0-9]+]] = hl.const #true
    // CHECK:   hl.cond.yield [[V1]]
    // CHECK: } incr {
    // CHECK: } do {
    // CHECK: }
    for (;;) {}
}

// CHECK-LABEL: hl.func @_Z11loop_nestedv
void loop_nested()
{
    // CHECK: hl.var @i : !hl.lvalue<!hl.int>
    // CHECK: hl.for {
    // CHECK:   [[V2:%[0-9]+]] = hl.cmp slt
    // CHECK:   hl.cond.yield [[V2]]
    // CHECK: } incr {
    // CHECK:   hl.pre.inc
    // CHECK: } do {
    for (int i = 0; i < 100; ++i) {
        // CHECK: hl.var @j : !hl.lvalue<!hl.int>
        // CHECK: hl.for {
        // CHECK:   [[V4:%[0-9]+]] = hl.cmp slt
        // CHECK:   hl.cond.yield [[V4]]
        // CHECK: } incr {
        // CHECK:   hl.pre.inc
        // CHECK: } do {
        for (int j = 0; j < 100; ++j) {

        }
        // CHECK: }
    }
    // CHECK: }
}
