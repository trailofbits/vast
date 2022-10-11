// RUN: vast-cc --from-source %s | vast-opt --vast-hl-lower-types --vast-hl-structs-to-tuples --vast-hl-to-scf | FileCheck %s

void fn()
{
    // CHECK: [[V0:%[0-9]+]] = hl.var "x" : !hl.lvalue<si32> = {
    //
    // CHECK: [[V1:%[0-9]+]] = hl.ref [[V0]] : !hl.lvalue<si32>
    // CHECK: [[V2:%[0-9]+]] = hl.implicit_cast [[V1]] LValueToRValue : !hl.lvalue<si32> -> si32
    // CHECK: [[V3:%[0-9]+]] = hl.implicit_cast [[V2]] IntegralToBoolean : si32 -> ui1
    // CHECK: [[V4:%[0-9]+]] = hl.implicit_cast [[V3]] IntegralCast : ui1 -> i1
    // CHECK: scf.if [[V4]] {
    // CHECK:   [[V5:%[0-9]+]] = hl.var "y" : !hl.lvalue<si32> = {
    // CHECK:     [[V6:%[0-9]+]] = hl.const #hl.integer<45> : si32
    // CHECK:     hl.value.yield [[V6]] : si32
    // CHECK:   }
    // CHECK: }
    // CHECK: hl.return
    int x = 12;
    if (x)
    {
        int y = 45;
    }
}
