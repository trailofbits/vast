// RUN: vast-cc --from-source %s | FileCheck %s

// CHECK-LABEL: func private @branch_ret
int branch_ret(int a, int b)
{
    // CHECK: [[V1:%[0-9]+]] = hl.cmp slt
    // CHECK-NEXT: hl.if [[V1]]
    if (a < b) {
        // CHECK: [[V2:%[0-9]+]] = hl.constant( 0 : i32 )
        // CHECK-NEXT: hl.return [[V2]]
        return 0;
    // CHECK: } else {
    } else {
        // CHECK: [[V3:%[0-9]+]] = hl.constant( 1 : i32 )
        // CHECK-NEXT: hl.return [[V3]]
        return 1;
    }
    // CHECK: hl.scope.end
}

// CHECK-LABEL: func private @branch_then
int branch_then(int a, int b)
{
    // CHECK: [[V1:%[0-9]+]] = hl.cmp eq
    // CHECK-NEXT: hl.if [[V1]]
    if (a == b) {
        // CHECK: [[V2:%[0-9]+]] = hl.constant( 0 : i32 )
        // CHECK-NEXT: hl.return [[V2]]
        return 0;
    }
    // CHECK: [[V3:%[0-9]+]] = hl.constant( 1 : i32 )
    // CHECK-NEXT: hl.return [[V3]]
    return 1;
}

// CHECK-LABEL: func private @branch_then_noreturn
int branch_then_noreturn(int a, int b)
{
    // CHECK: [[V1:%[0-9]+]] = hl.cmp sgt
    // CHECK-NEXT: hl.if [[V1]]
    if (a > b) {
        // CHECK: [[V2:%[0-9]+]] = hl.addi
        // CHECK-NEXT: hl.var( c, [[V2]] )
        // CHECK-NEXT: hl.scope.end
        int c = a + b;
    }

    // CHECK: [[V3:%[0-9]+]] = hl.constant( 1 : i32 )
    // CHECK-NEXT: hl.return [[V3]]
    return 1;
}

// CHECK-LABEL: func private @branch_then_empty
int branch_then_empty(int a, int b)
{
    // CHECK: [[V1:%[0-9]+]] = hl.cmp ne
    // CHECK-NEXT: hl.if [[V1]]
    if (a != b) {
        // CHECK-NEXT: hl.scope.end
    }
    // CHECK: [[V3:%[0-9]+]] = hl.constant( 1 : i32 )
    // CHECK-NEXT: hl.return [[V3]]
    return 1;
}

// CHECK-LABEL: func private @branch_else_empty
int branch_else_empty(int a, int b)
{
    // CHECK: [[V1:%[0-9]+]] = hl.cmp sle
    // CHECK-NEXT: hl.if [[V1]]
    if (a <= b) {
        // CHECK: [[V2:%[0-9]+]] = hl.constant
        // CHECK-NEXT: hl.var( c, [[V2]] )
        int c = 7;
        // CHECK-NEXT: hl.scope.end
        // CHECK-NEXT: } else {
    } else {
        // CHECK-NEXT: hl.scope.end
    }
    // CHECK: [[V3:%[0-9]+]] = hl.constant( 1 : i32 )
    // CHECK-NEXT: hl.return [[V3]]
    return 1;
}

// CHECK-LABEL: func private @branch_empty
int branch_empty(int a, int b)
{
    // CHECK: [[V1:%[0-9]+]] = hl.cmp sge
    // CHECK-NEXT: hl.if [[V1]]
    if (a >= b) {
        // CHECK-NEXT: hl.scope.end
        // CHECK-NEXT: } else {
    } else {
        // CHECK-NEXT: hl.scope.end
    }
    // CHECK: [[V3:%[0-9]+]] = hl.constant( 1 : i32 )
    // CHECK-NEXT: hl.return [[V3]]
    return 1;
}
