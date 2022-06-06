// RUN: vast-cc --from-source %s | vast-opt --vast-hl-lower-types --vast-hl-structs-to-tuples --vast-hl-to-scf | FileCheck %s

void fn()
{
    // CHECK: [[V0:%[0-9]+]] = hl.var "x" : !hl.lvalue<i32> = {
    //
    // CHECK: [[V1:%[0-9]+]] = hl.decl.ref [[V0]] : !hl.lvalue<i32>
    // CHECK: [[V2:%[0-9]+]] = hl.implicit_cast [[V1]] LValueToRValue : !hl.lvalue<i32> -> i32
    // CHECK: [[V3:%[0-9]+]] = hl.implicit_cast [[V2]] IntegralToBoolean : i32 -> i1
    // CHECK: scf.if [[V3]] {
    // CHECK:   [[V4:%[0-9]+]] = hl.var "y" : !hl.lvalue<i32> = {
    // CHECK:     [[V5:%[0-9]+]] = hl.constant.int 45 : i32
    // CHECK:     hl.value.yield [[V5]] : i32
    // CHECK:   }
    // CHECK: }
    // CHECK: hl.return
    int x = 12;
    if (x)
    {
        int y = 45;
    }
}
