// RUN: vast-cc --ccopts -xc --from-source %s | vast-opt --vast-hl-lower-types --vast-hl-structs-to-tuples --vast-hl-to-scf | FileCheck %s

int fn()
{
    int x = 12;
    // CHECK: scf.if [[V3:%[0-9]+]] {
    // CHECK:   [[V5:%[0-9]+]] = hl.ref [[V0:%[0=9]+]] : !hl.lvalue<si32>
    // CHECK:   [[V6:%[0-9]+]] = hl.implicit_cast [[V5]] LValueToRValue : !hl.lvalue<si32> -> si32
    // CHECK:   hl.scope {
    // CHECK:     hl.return [[V6]] : si32
    // CHECK:   }
    // CHECK: } else {
    // CHECK:   [[V7:%[0-9]+]] = hl.const #hl.integer<5> : si32
    // CHECK:   hl.scope {
    // CHECK:     hl.return [[V7]] : si32
    // CHECK:   }
    // CHECK: }
    if (x)
    {
        return x;
    } else {
        return 5;
    }
    return 1;
}
