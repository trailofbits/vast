// RUN: vast-cc --ccopts -xc --from-source %s | vast-opt --vast-hl-lower-types | FileCheck %s

void loop_simple()
{
    // CHECK: [[I:%[0-9]+]] = hl.var "i" : !hl.lvalue<si32>
    // CHECK: hl.for
    // CHECK:   [[V0:%[0-9]+]] = hl.ref [[I]]
    // CHECK:   [[V1:%[0-9]+]] = hl.implicit_cast [[V0]] LValueToRValue : !hl.lvalue<si32> -> si32
    // CHECK:   [[V2:%[0-9]+]] = hl.const #hl.integer<100> : si32
    // CHECK:   [[V3:%[0-9]+]] = hl.cmp slt [[V1]], [[V2]] : si32, si32 -> si32
    // CHECK:   hl.cond.yield [[V3]] : si32
    for (int i = 0; i < 100; ++i) {
        /* ... */
    }
}
