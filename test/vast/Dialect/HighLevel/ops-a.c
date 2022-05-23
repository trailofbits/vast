// RUN: vast-cc --from-source %s | FileCheck %s
// RUN: vast-cc --from-source %s > %t && vast-opt %t | diff -B %t -

// CHECK: func @add1([[A1:%arg[0-9]+]]: !hl.lvalue<!hl.int>, [[A2:%arg[0-9]+]]: !hl.lvalue<!hl.int>) -> !hl.int
int add1(int a, int b)
{
    // CHECK: [[V1:%[0-9]+]] = hl.decl.ref [[A1]] : !hl.lvalue<!hl.int>
    // CHECK: [[V2:%[0-9]+]] = hl.implicit_cast [[V1]] LValueToRValue : !hl.lvalue<!hl.int> -> !hl.int
    // CHECK: [[V3:%[0-9]+]] = hl.decl.ref [[A2]] : !hl.lvalue<!hl.int>
    // CHECK: [[V4:%[0-9]+]] = hl.implicit_cast [[V3]] LValueToRValue : !hl.lvalue<!hl.int> -> !hl.int
    // CHECK: hl.add [[V2]], [[V4]] : !hl.int
    return a + b;
}

// CHECK: func @add2([[A1:%arg[0-9]+]]: !hl.lvalue<!hl.int>, [[A2:%arg[0-9]+]]: !hl.lvalue<!hl.int>) -> !hl.int
int add2(int a, int b)
{
    // CHECK: [[R:%[0-9]+]] = hl.var "r" : !hl.lvalue<!hl.int> = {
    // CHECK:   [[V1:%[0-9]+]] = hl.decl.ref [[A1]] : !hl.lvalue<!hl.int>
    // CHECK:   [[V2:%[0-9]+]] = hl.implicit_cast [[V1]] LValueToRValue : !hl.lvalue<!hl.int> -> !hl.int
    // CHECK:   [[V3:%[0-9]+]] = hl.decl.ref [[A2]] : !hl.lvalue<!hl.int>
    // CHECK:   [[V4:%[0-9]+]] = hl.implicit_cast [[V3]] LValueToRValue : !hl.lvalue<!hl.int> -> !hl.int
    // CHECK:   [[V5:%[0-9]+]] = hl.add [[V2]], [[V4]] : !hl.int
    // CHECK:   hl.value.yield [[V5]]
    int r = a + b;

    // CHECK: [[V7:%[0-9]+]] = hl.decl.ref [[R]] : !hl.lvalue<!hl.int>
    // CHECK: [[V8:%[0-9]+]] = hl.implicit_cast [[V7]] LValueToRValue : !hl.lvalue<!hl.int> -> !hl.int
    // CHECK: return [[V8]] : !hl.int
    return r;
}

// CHECK: func @add3() -> !hl.void
void add3()
{
    // CHECK: hl.var "v" : !hl.lvalue<!hl.int> = {
    // CHECK:   [[V1:%[0-9]+]] = hl.constant.int 1 : !hl.int
    // CHECK:   [[V2:%[0-9]+]] = hl.constant.int 2 : !hl.int
    // CHECK:   [[V3:%[0-9]+]] = hl.add [[V1]], [[V2]] : !hl.int
    // CHECK:   hl.value.yield [[V3]]
    int v = 1 + 2;
}
