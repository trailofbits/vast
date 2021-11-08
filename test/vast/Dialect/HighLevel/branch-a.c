// RUN: vast-cc --from-source %s | FileCheck %s

// CHECK-LABEL: func private @branch_ret
int branch_ret(int a, int b)
{
    // CHECK: hl.if {
    // CHECK: [[V1:%[0-9]+]] = hl.cmp slt
    // CHECK: hl.cond.yield [[V1]]
    if (a < b) {
        // CHECK: } then {
        // CHECK: [[V2:%[0-9]+]] = hl.constant 0
        // CHECK: hl.return [[V2]]
        return 0;
    } else {
        // CHECK: } else {
        // CHECK: [[V3:%[0-9]+]] = hl.constant 1
        // CHECK: hl.return [[V3]]
        return 1;
    }
    // CHECK: hl.unreachable
}

// CHECK-LABEL: func private @branch_then
int branch_then(int a, int b)
{
    // CHECK: hl.if
    // CHECK: [[V1:%[0-9]+]] = hl.cmp eq
    // CHECK: hl.cond.yield [[V1]]
    if (a == b) {
        // CHECK: } then {
        // CHECK: [[V2:%[0-9]+]] = hl.constant 0
        // CHECK-NEXT: hl.return [[V2]]
        return 0;
    }
    // CHECK-NOT: else
    // CHECK: [[V3:%[0-9]+]] = hl.constant 1
    // CHECK-NEXT: hl.return [[V3]]
    return 1;
}

// CHECK-LABEL: func private @branch_then_noreturn
int branch_then_noreturn(int a, int b)
{
    // CHECK: hl.if
    // CHECK: [[V1:%[0-9]+]] = hl.cmp sgt
    // CHECK: hl.cond.yield [[V1]]
    if (a > b) {
        // CHECK: } then {
        // CHECK: [[V2:%[0-9]+]] = hl.add
        // CHECK: hl.var @c = [[V2]]
        int c = a + b;
    }

    // CHECK-NOT: else
    // CHECK: [[V3:%[0-9]+]] = hl.constant 1
    // CHECK-NEXT: hl.return [[V3]]
    return 1;
}

// CHECK-LABEL: func private @branch_then_empty
int branch_then_empty(int a, int b)
{
    // CHECK: hl.if
    // CHECK: [[V1:%[0-9]+]] = hl.cmp ne
    // CHECK: hl.cond.yield [[V1]]
    if (a != b) {
        // CHECK: } then {
    }
    // CHECK-NOT: else
    // CHECK: [[V3:%[0-9]+]] = hl.constant 1
    // CHECK-NEXT: hl.return [[V3]]
    return 1;
}

// CHECK-LABEL: func private @branch_else_empty
int branch_else_empty(int a, int b)
{
    // CHECK: hl.if
    // CHECK: [[V1:%[0-9]+]] = hl.cmp sle
    // CHECK: hl.cond.yield [[V1]]
    if (a <= b) {
        // CHECK: } then {
        // CHECK: [[V2:%[0-9]+]] = hl.constant
        // CHECK-NEXT: hl.var @c = [[V2]]
        int c = 7;
    } else {
        // CHECK: } else {
    }
    // CHECK: [[V3:%[0-9]+]] = hl.constant 1
    // CHECK-NEXT: hl.return [[V3]]
    return 1;
}

// CHECK-LABEL: func private @branch_empty
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
    // CHECK: [[V2:%[0-9]+]] = hl.constant 1
    // CHECK-NEXT: hl.return [[V2]]
    return 1;
}

// CHECK-LABEL: func private @branch_true
int branch_true(int a, int b)
{
    // CHECK: hl.if
    // CHECK: [[V1:%[0-9]+]] = hl.constant true
    // CHECK: hl.cond.yield [[V1]]
    if (true) {
        // CHECK: } then {
    }
    // CHECK: [[V2:%[0-9]+]] = hl.constant 1
    // CHECK-NEXT: hl.return [[V2]]
    return 1;
}
