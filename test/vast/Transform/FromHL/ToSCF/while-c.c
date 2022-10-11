// RUN: vast-cc --ccopts -xc --from-source %s | vast-opt --vast-hl-lower-types --vast-hl-structs-to-tuples --vast-hl-to-scf | FileCheck %s

int fn()
{
    int x = 5;

    // CHECK: scf.while : () -> () {
    // CHECK:   [[V2:%[0-9]+]] = hl.ref [[V0:%[0-9]+]] : !hl.lvalue<si32>
    // CHECK:   [[V3:%[0-9]+]] = hl.implicit_cast [[V2]] LValueToRValue : !hl.lvalue<si32> -> si32
    // CHECK:   [[V4:%[0-9]+]] = hl.const #hl.integer<0> : si32
    // CHECK:   [[V5:%[0-9]+]] = hl.cmp ne [[V3]], [[V4]] : si32, si32 -> si32
    // CHECK:   [[V6:%[0-9]+]] = hl.implicit_cast [[V5]] IntegralCast : si32 -> i1
    // CHECK:   scf.condition([[V6]])
    // CHECK: } do {
    // CHECK:   [[V12:%[0-9]+]] = hl.ref [[V0]] : !hl.lvalue<si32>
    // CHECK:   [[V13:%[0-9]+]] = hl.implicit_cast [[V12]] LValueToRValue : !hl.lvalue<si32> -> si32
    // CHECK:   hl.scope {
    // CHECK:     hl.return [[V13]] : si32
    // CHECK:   }
    // CHECK:   scf.yield
    // CHECK: }

    while (x != 0)
        return x;
    return 0;
}
