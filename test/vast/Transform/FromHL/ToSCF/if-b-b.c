// RUN: vast-cc --ccopts -xc --from-source %s | vast-opt --vast-hl-lower-types --vast-hl-structs-to-tuples --vast-hl-lower-cf | FileCheck %s

void fn()
{
    int x = 12;

    // CHECK: [[V1:%[0-9]+]] = hl.decl.ref [[V0]] : !hl.lvalue<i32>
    // CHECK: [[V2:%[0-9]+]] = hl.implicit_cast [[V1]] LValueToRValue : !hl.lvalue<i32> -> i32
    // CHECK: [[V3:%[0-9]+]] = hl.implicit_cast [[V2]] IntegralCast : i32 -> i1
    // CHECK: scf.if [[V3]] {
    // CHECK:   [[V14:%[0-9]+]] = hl.var "y" : !hl.lvalue<i32> = {
    // CHECK:     [[V15:%[0-9]+]] = hl.constant.int 45 : i32
    // CHECK:     hl.value.yield [[V15]] : i32
    // CHECK:   }
    // CHECK: } else {
    // CHECK:   [[V24:%[0-9]+]] = hl.var "y" : !hl.lvalue<i32> = {
    // CHECK:     [[V25:%[0-9]+]] = hl.constant.int 55 : i32
    // CHECK:     hl.value.yield [[V25]] : i32
    // CHECK:   }
    // CHECK: }
    // CHECK: hl.return

    if (x)
    {
        int y = 45;
    } else {
        int y = 55;
    }
}
