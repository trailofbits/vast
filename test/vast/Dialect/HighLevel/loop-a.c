// RUN: vast-cc --from-source %s | FileCheck %s

// CHECK-LABEL: func private @loop_simple
void loop_simple()
{
    // CHECK: hl.for {
    // CHECK: hl.var( i, [[V1:%[0-9]+]] )
    // CHECK: } cond {
    // CHECK: hl.cmp slt
    // CHECK: } incr {
    // CHECK: hl.post.inc
    // CHECK: } do {
    // CHECK: }
    for (int i = 0; i < 100; i++) {}
}

// CHECK-LABEL: func private @loop_noinit
void loop_noinit()
{
    int i = 0;
    // CHECK: hl.var( i, [[V1:%[0-9]+]] )
    // CHECK: hl.for {
    // CHECK: } cond {
    // CHECK: [[V2:%[0-9]+]] = hl.cmp slt
    // CHECK: hl.cond.yield [[V2]]
    // CHECK: } incr {
    // CHECK: hl.pre.inc
    // CHECK: } do {
    // CHECK: }
    for (;i < 100; ++i) {}
}

// CHECK-LABEL: func private @loop_noincr
void loop_noincr()
{
    // CHECK: hl.for {
    // CHECK: hl.var( i, [[V1:%[0-9]+]] )
    // CHECK: } cond {
    // CHECK: [[V2:%[0-9]+]] = hl.cmp slt
    // CHECK: hl.cond.yield [[V2]]
    // CHECK: } incr {
    // CHECK: } do {
    // CHECK: hl.pre.inc
    // CHECK: }
    for (int i = 0; i < 100;) { ++i; }
}

// CHECK-LABEL: func private @loop_infinite
void loop_infinite()
{
    // CHECK: hl.for {
    // CHECK: } cond {
    // CHECK: [[V1:%[0-9]+]] = hl.constant( true )
    // CHECK: hl.cond.yield [[V1]]
    // CHECK: } incr {
    // CHECK: } do {
    // CHECK: }
    for (;;) {}
}

// CHECK-LABEL: func private @loop_nested
void loop_nested()
{
    // CHECK: hl.for {
    // CHECK: hl.var( i, [[V1:%[0-9]+]] )
    // CHECK: } cond {
    // CHECK: [[V2:%[0-9]+]] = hl.cmp slt
    // CHECK: hl.cond.yield [[V2]]
    // CHECK: } incr {
    // CHECK: hl.pre.inc
    // CHECK: } do {
    for (int i = 0; i < 100; ++i) {
        // CHECK: hl.for {
        // CHECK: hl.var( j, [[V3:%[0-9]+]] )
        // CHECK: } cond {
        // CHECK: [[V4:%[0-9]+]] = hl.cmp slt
        // CHECK: hl.cond.yield [[V4]]
        // CHECK: } incr {
        // CHECK: hl.pre.inc
        // CHECK: } do {
        for (int j = 0; j < 100; ++j) {

        }
        // CHECK: }
    }
    // CHECK: }
}
