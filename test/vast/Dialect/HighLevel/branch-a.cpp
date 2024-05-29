// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

// CHECK-LABEL: hl.func @_Z10branch_retii
int branch_ret(int a, int b)
{
    // CHECK: hl.if {
    // CHECK: [[V1:%[0-9]+]] = hl.cmp slt
    // CHECK: hl.cond.yield [[V1]]
    if (a < b) {
        // CHECK: } then {
        // CHECK: [[V2:%[0-9]+]] = hl.const #core.integer<0> : !hl.int
        // CHECK: hl.return [[V2]]
        return 0;
    } else {
        // CHECK: } else {
        // CHECK: [[V3:%[0-9]+]] = hl.const #core.integer<1> : !hl.int
        // CHECK: hl.return [[V3]]
        return 1;
    }
}

// CHECK-LABEL: hl.func @_Z11branch_thenii
int branch_then(int a, int b)
{
    // CHECK: hl.if
    // CHECK: [[V1:%[0-9]+]] = hl.cmp eq
    // CHECK: hl.cond.yield [[V1]]
    if (a == b) {
        // CHECK: } then {
        // CHECK: [[V2:%[0-9]+]] = hl.const #core.integer<0> : !hl.int
        // CHECK-NEXT: hl.return [[V2]]
        return 0;
    }
    // CHECK-NOT: else
    // CHECK: [[V3:%[0-9]+]] = hl.const #core.integer<1> : !hl.int
    // CHECK-NEXT: hl.return [[V3]]
    return 1;
}

// CHECK-LABEL: hl.func @_Z20branch_then_noreturnii
int branch_then_noreturn(int a, int b)
{
    // CHECK: hl.if
    // CHECK: [[V1:%[0-9]+]] = hl.cmp sgt
    // CHECK: hl.cond.yield [[V1]]
    if (a > b) {
        // CHECK: } then {
        // CHECK: hl.var @c : !hl.lvalue<!hl.int> =
        // CHECK: [[V2:%[0-9]+]] = hl.add
        // CHECK: hl.value.yield [[V2]]
        int c = a + b;
    }

    // CHECK-NOT: else
    // CHECK: [[V3:%[0-9]+]] = hl.const #core.integer<1> : !hl.int
    // CHECK-NEXT: hl.return [[V3]]
    return 1;
}

// CHECK-LABEL: hl.func @_Z17branch_then_emptyii
int branch_then_empty(int a, int b)
{
    // CHECK: hl.if
    // CHECK: [[V1:%[0-9]+]] = hl.cmp ne
    // CHECK: hl.cond.yield [[V1]]
    if (a != b) {
        // CHECK: } then {
    }
    // CHECK-NOT: else
    // CHECK: [[V3:%[0-9]+]] = hl.const #core.integer<1> : !hl.int
    // CHECK-NEXT: hl.return [[V3]]
    return 1;
}

// CHECK-LABEL: hl.func @_Z17branch_else_emptyii
int branch_else_empty(int a, int b)
{
    // CHECK: hl.if
    // CHECK: [[V1:%[0-9]+]] = hl.cmp sle
    // CHECK: hl.cond.yield [[V1]]
    if (a <= b) {
        // CHECK: } then {
        // CHECK-NEXT: hl.var @c : !hl.lvalue<!hl.int> =
        // CHECK: [[V2:%[0-9]+]] = hl.const #core.integer<7> : !hl.int
        // CHECK: hl.value.yield [[V2]]
        int c = 7;
    } else {
        // CHECK: } else {
    }
    // CHECK: [[V3:%[0-9]+]] = hl.const #core.integer<1> : !hl.int
    // CHECK-NEXT: hl.return [[V3]]
    return 1;
}

// CHECK-LABEL: hl.func @_Z12branch_emptyii
int branch_empty(int a, int b)
{
    // CHECK: hl.if
    // CHECK: [[V1:%[0-9]+]] = hl.cmp sge
    // CHECK: hl.cond.yield [[V1]]
    if (a >= b) {
        // CHECK: } then {
    } else {
        // CHECK: } else {
    }
    // CHECK: [[V2:%[0-9]+]] = hl.const #core.integer<1> : !hl.int
    // CHECK-NEXT: hl.return [[V2]]
    return 1;
}

// CHECK-LABEL: hl.func @_Z11branch_trueii
int branch_true(int a, int b)
{
    // CHECK: hl.if
    // CHECK: [[V1:%[0-9]+]] = hl.const #true
    // CHECK: hl.cond.yield [[V1]]
    if (true) {
        // CHECK: } then {
    }
    // CHECK: [[V2:%[0-9]+]] = hl.const #core.integer<1> : !hl.int
    // CHECK-NEXT: hl.return [[V2]]
    return 1;
}
