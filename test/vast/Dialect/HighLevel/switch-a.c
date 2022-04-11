// RUN: vast-cc --from-source %s | FileCheck %s
// RUN: vast-cc --from-source %s > %t && vast-opt %t | diff -B %t -

// CHECK: func @switch_simple([[A1:%arg[0-9]+]]: !hl.lvalue<!hl.int>) -> !hl.int
int switch_simple(int num)
{
    // CHECK: hl.switch {
    // CHECK: [[V1:%[0-9]+]] = hl.declref [[A1]]
    // CHECK: [[V2:%[0-9]+]] = hl.implicit_cast [[V1]]
    // CHECK: hl.value.yield [[V2]]
    switch (num) {
        // CHECK: } cases {
        case  1: return 1;
        // CHECK: hl.case {
        // CHECK: [[V3:%[0-9]+]] = hl.constant.int 1 : !hl.int
        // CHECK: hl.value.yield [[V3]]
        // CHECK: } {
        // CHECK: [[V4:%[0-9]+]] = hl.constant.int 1 : !hl.int
        // CHECK: hl.return [[V4]]
        // CHECK: }
        case  2: return 2;
        // CHECK: hl.case {
        // CHECK: [[V5:%[0-9]+]] = hl.constant.int 2 : !hl.int
        // CHECK: hl.value.yield [[V5]]
        // CHECK: } {
        // CHECK: [[V6:%[0-9]+]] = hl.constant.int 2 : !hl.int
        // CHECK: hl.return [[V6]]
        // CHECK: }
        default: return 0;
        // CHECK: hl.default {
        // CHECK: [[V7:%[0-9]+]] = hl.constant.int 0 : !hl.int
        // CHECK: hl.return [[V7]]
        // CHECK: }
    }
    // CHECK: }
    // CHECK: hl.unreachable
}

// CHECK: func @switch_fallthorugh_1([[A1:%arg[0-9]+]]: !hl.lvalue<!hl.int>) -> !hl.int
int switch_fallthorugh_1(int num)
{
    // CHECK: hl.switch {
    // CHECK: [[V1:%[0-9]+]] = hl.declref [[A1]]
    // CHECK: [[V2:%[0-9]+]] = hl.implicit_cast [[V1]]
    // CHECK: hl.value.yield [[V2]]
    switch (num) {
        // CHECK: } cases {
        // CHECK: hl.case {
        case  1:
        // CHECK: hl.case {
        case  2:
        // CHECK: hl.default {
        default: return 0;
        // CHECK: }
        // CHECK: }
        // CHECK: }
    }
    // CHECK: }
    // CHECK: hl.unreachable
}

// CHECK: func @switch_fallthorugh_2([[A1:%arg[0-9]+]]: !hl.lvalue<!hl.int>) -> !hl.int
int switch_fallthorugh_2(int num)
{
    // CHECK: hl.switch {
    // CHECK: [[V1:%[0-9]+]] = hl.declref [[A1]]
    // CHECK: [[V2:%[0-9]+]] = hl.implicit_cast [[V1]]
    // CHECK: hl.value.yield [[V2]]
    switch (num) {
        // CHECK: } cases {
        // CHECK: hl.case {
        case  1:
        // CHECK: hl.case {
        // CHECK: }
        // CHECK: }
        case  2: return 1;
        // CHECK: hl.default {
        default: return 0;
        // CHECK: }
    }
    // CHECK: }
    // CHECK: hl.unreachable
}

// CHECK: func @switch_nodefault([[A1:%arg[0-9]+]]: !hl.lvalue<!hl.int>) -> !hl.int
int switch_nodefault(int num)
{
    // CHECK: hl.switch {
    // CHECK: [[V1:%[0-9]+]] = hl.declref [[A1]]
    // CHECK: [[V2:%[0-9]+]] = hl.implicit_cast [[V1]]
    // CHECK: hl.value.yield [[V2]]
    switch (num) {
        // CHECK: } cases {
        // CHECK: hl.case {
        case  1:
        // CHECK: hl.case {
        case  2: return 1;
        // CHECK: }
        // CHECK: }
    }
    // CHECK: }

    return 0;
}

// CHECK: func @switch_break([[A1:%arg[0-9]+]]: !hl.lvalue<!hl.int>) -> !hl.int
int switch_break(int num)
{
    // CHECK: hl.switch {
    // CHECK: [[V1:%[0-9]+]] = hl.declref [[A1]]
    // CHECK: [[V2:%[0-9]+]] = hl.implicit_cast [[V1]]
    // CHECK: hl.value.yield [[V2]]
    switch (num) {
        // CHECK: } cases {
        // CHECK: hl.case {
        // CHECK: hl.break
        // CHECK: }
        case  1: break;
        // CHECK: hl.case {
        // CHECK: }
        case  2: return 1;
    }
    // CHECK: }

    return 0;
}

// CHECK: func @switch_block([[A1:%arg[0-9]+]]: !hl.lvalue<!hl.int>) -> !hl.int
int switch_block(int num)
{
    // CHECK: hl.switch {
    // CHECK: [[V1:%[0-9]+]] = hl.declref [[A1]]
    // CHECK: [[V2:%[0-9]+]] = hl.implicit_cast [[V1]]
    // CHECK: hl.value.yield [[V2]]
    switch (num) {
        // CHECK: } cases {
        // CHECK: hl.case {
        // CHECK: hl.var
        // CHECK: }
        case  1: {
            int x = 0;
        }
        // CHECK: hl.case {
        // CHECK: hl.return
        // CHECK: }
        case  2: return 1;
    }
    // CHECK: }

    return 0;
}

// CHECK: func @switch_single([[A1:%arg[0-9]+]]: !hl.lvalue<!hl.int>) -> !hl.void
void switch_single(int num)
{
    int v = 0;
    // CHECK: hl.switch {
    // CHECK: [[V1:%[0-9]+]] = hl.declref [[A1]]
    // CHECK: [[V2:%[0-9]+]] = hl.implicit_cast [[V1]]
    // CHECK: hl.value.yield [[V2]]
    switch (num)
        // CHECK: } cases {
        // CHECK: hl.declref
        // CHECK: hl.post.inc
        // CHECK: }
        v++;
}

// CHECK: func @switch_no_compound([[A1:%arg[0-9]+]]: !hl.lvalue<!hl.int>) -> !hl.void
void switch_no_compound(int num)
{
    int v = 0;
    // CHECK: hl.switch {
    // CHECK: [[V1:%[0-9]+]] = hl.declref [[A1]]
    // CHECK: [[V2:%[0-9]+]] = hl.implicit_cast [[V1]]
    // CHECK: hl.value.yield [[V2]]
    switch (num)
        // CHECK: } cases {
        // CHECK: hl.case {
        // CHECK: hl.case {
        // CHECK: hl.declref
        // CHECK: hl.post.inc
        case 0:
        case 1: v++;
        // CHECK: }
        // CHECK: }
}
