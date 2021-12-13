// RUN: vast-cc --ccopts -std=c++17 --from-source %s | FileCheck %s
// RUN: vast-cc --ccopts -std=c++17 --from-source %s > %t && vast-opt %t | diff -B %t -

// CHECK-LABEL: func private @switch_simple
int switch_simple(int num)
{
    // CHECK: hl.switch cond {
    // CHECK: [[V1:%[0-9]+]] = hl.declref @num
    // CHECK: [[V2:%[0-9]+]] = hl.implicit_cast [[V1]]
    // CHECK: hl.value.yield [[V2]]
    switch (num) {
        // CHECK: } cases {
        case  1: return 1;
        // CHECK: hl.case {
        // CHECK: [[V3:%[0-9]+]] = hl.constant(#hl.int<1>)
        // CHECK: hl.value.yield [[V3]]
        // CHECK: } {
        // CHECK: [[V4:%[0-9]+]] = hl.constant(#hl.int<1>)
        // CHECK: hl.return [[V4]]
        // CHECK: }
        case  2: return 2;
        // CHECK: hl.case {
        // CHECK: [[V5:%[0-9]+]] = hl.constant(#hl.int<2>)
        // CHECK: hl.value.yield [[V5]]
        // CHECK: } {
        // CHECK: [[V6:%[0-9]+]] = hl.constant(#hl.int<2>)
        // CHECK: hl.return [[V6]]
        // CHECK: }
        default: return 0;
        // CHECK: hl.default {
        // CHECK: [[V7:%[0-9]+]] = hl.constant(#hl.int<0>)
        // CHECK: hl.return [[V7]]
        // CHECK: }
    }
    // CHECK: }
    // CHECK: hl.unreachable
}

// CHECK-LABEL: func private @switch_init
int switch_init(int num)
{
    // CHECK: hl.switch init {
    // CHECK: hl.var @v : !hl.int
    // CHECK: } cond {
    // CHECK: [[V2:%[0-9]+]] = hl.declref @v
    // CHECK: [[V3:%[0-9]+]] = hl.implicit_cast [[V2]]
    // CHECK: hl.value.yield [[V3]]
    switch (int v = num; v) {
        // CHECK: } cases {
        case  1: return 1;
        case  2: return 2;
        default: return 0;
    }
    // CHECK: }
    // CHECK: hl.unreachable
}

// CHECK-LABEL: func private @switch_fallthorugh_1
int switch_fallthorugh_1(int num)
{
    // CHECK: hl.switch cond {
    // CHECK: [[V1:%[0-9]+]] = hl.declref @num
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

// CHECK-LABEL: func private @switch_fallthorugh_2
int switch_fallthorugh_2(int num)
{
    // CHECK: hl.switch cond {
    // CHECK: [[V1:%[0-9]+]] = hl.declref @num
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

// CHECK-LABEL: func private @switch_nodefault
int switch_nodefault(int num)
{
    // CHECK: hl.switch cond {
    // CHECK: [[V1:%[0-9]+]] = hl.declref @num
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

// CHECK-LABEL: func private @switch_break
int switch_break(int num)
{
    // CHECK: hl.switch cond {
    // CHECK: [[V1:%[0-9]+]] = hl.declref @num
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

// CHECK-LABEL: func private @switch_block
int switch_block(int num)
{
    // CHECK: hl.switch cond {
    // CHECK: [[V1:%[0-9]+]] = hl.declref @num
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

// CHECK-LABEL: func private @switch_single
void switch_single(int num)
{
    int v = 0;
    // CHECK: hl.switch cond {
    // CHECK: [[V1:%[0-9]+]] = hl.declref @num
    // CHECK: [[V2:%[0-9]+]] = hl.implicit_cast [[V1]]
    // CHECK: hl.value.yield [[V2]]
    switch (num)
        // CHECK: } cases {
        // CHECK: hl.declref
        // CHECK: hl.post.inc
        // CHECK: }
        v++;
}

// CHECK-LABEL: func private @switch_no_compound
void switch_no_compound(int num)
{
    int v = 0;
    // CHECK: hl.switch cond {
    // CHECK: [[V1:%[0-9]+]] = hl.declref @num
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
