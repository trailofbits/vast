// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

// CHECK: hl.func @add1 {{.*}} ([[A1:%arg[0-9]+]]: !hl.lvalue<!hl.int>, [[A2:%arg[0-9]+]]: !hl.lvalue<!hl.int>) -> !hl.int
int add1(int a, int b)
{
    // CHECK: [[V1:%[0-9]+]] = hl.ref @a
    // CHECK: [[V2:%[0-9]+]] = hl.implicit_cast [[V1]] LValueToRValue : !hl.lvalue<!hl.int> -> !hl.int
    // CHECK: [[V3:%[0-9]+]] = hl.ref @b
    // CHECK: [[V4:%[0-9]+]] = hl.implicit_cast [[V3]] LValueToRValue : !hl.lvalue<!hl.int> -> !hl.int
    // CHECK: hl.add [[V2]], [[V4]] : (!hl.int, !hl.int) -> !hl.int
    return a + b;
}

// CHECK: hl.func @add2 {{.*}} ([[A1:%arg[0-9]+]]: !hl.lvalue<!hl.int>, [[A2:%arg[0-9]+]]: !hl.lvalue<!hl.int>) -> !hl.int
int add2(int a, int b)
{
    // CHECK: hl.var @r : !hl.lvalue<!hl.int> = {
    // CHECK:   [[V1:%[0-9]+]] = hl.ref @a
    // CHECK:   [[V2:%[0-9]+]] = hl.implicit_cast [[V1]] LValueToRValue : !hl.lvalue<!hl.int> -> !hl.int
    // CHECK:   [[V3:%[0-9]+]] = hl.ref @b
    // CHECK:   [[V4:%[0-9]+]] = hl.implicit_cast [[V3]] LValueToRValue : !hl.lvalue<!hl.int> -> !hl.int
    // CHECK:   [[V5:%[0-9]+]] = hl.add [[V2]], [[V4]] : (!hl.int, !hl.int) -> !hl.int
    // CHECK:   hl.value.yield [[V5]]
    int r = a + b;

    // CHECK: [[V7:%[0-9]+]] = hl.ref @r
    // CHECK: [[V8:%[0-9]+]] = hl.implicit_cast [[V7]] LValueToRValue : !hl.lvalue<!hl.int> -> !hl.int
    // CHECK: return [[V8]] : !hl.int
    return r;
}

// CHECK: hl.func @add3
void add3()
{
    // CHECK: hl.var @v : !hl.lvalue<!hl.int> = {
    // CHECK:   [[V1:%[0-9]+]] = hl.const #core.integer<1> : !hl.int
    // CHECK:   [[V2:%[0-9]+]] = hl.const #core.integer<2> : !hl.int
    // CHECK:   [[V3:%[0-9]+]] = hl.add [[V1]], [[V2]] : (!hl.int, !hl.int) -> !hl.int
    // CHECK:   hl.value.yield [[V3]]
    int v = 1 + 2;
}
