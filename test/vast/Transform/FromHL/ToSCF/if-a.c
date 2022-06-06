// RUN: vast-cc --ccopts -xc --from-source %s | vast-opt --vast-hl-lower-types --vast-hl-structs-to-tuples --vast-hl-to-scf | FileCheck %s

void fn()
{
    // CHECK: [[V0:%[0-9]+]] = hl.var "x" : !hl.lvalue<i32> = {
    // CHECK:   [[V14:%[0-9]+]] = hl.constant.int 12 : i32
    // CHECK:   hl.value.yield [[V14]] : i32
    // CHECK: }
    // CHECK: [[V1:%[0-9]+]] = hl.decl.ref [[V0]] : !hl.lvalue<i32>
    // CHECK: [[V2:%[0-9]+]] = hl.implicit_cast [[V1]] LValueToRValue : !hl.lvalue<i32> -> i32
    // CHECK: [[V3:%[0-9]+]] = hl.implicit_cast [[V2]] IntegralCast : i32 -> i1
    // CHECK: scf.if [[V3]] {
    // CHECK:   [[V24:%[0-9]+]] = hl.var "y" : !hl.lvalue<i32> = {
    // CHECK:     [[V25:%[0-9]+]] = hl.constant.int 45 : i32
    // CHECK:     hl.value.yield [[V25]] : i32
    // CHECK:   }
    // CHECK: }
    // CHECK: hl.return

    int x = 12;
    if (x)
    {
        int y = 45;
    }
}
