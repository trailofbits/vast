// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %vast-opt --vast-hl-lower-types | %file-check %s

void loop_simple()
{
    // CHECK: hl.var @i : !hl.lvalue<si32>
    // CHECK: hl.for
    // CHECK:   [[V0:%[0-9]+]] = hl.ref @i
    // CHECK:   [[V1:%[0-9]+]] = hl.implicit_cast [[V0]] LValueToRValue : !hl.lvalue<si32> -> si32
    // CHECK:   [[V2:%[0-9]+]] = hl.const #core.integer<100> : si32
    // CHECK:   [[V3:%[0-9]+]] = hl.cmp slt [[V1]], [[V2]] : si32, si32 -> si32
    // CHECK:   hl.cond.yield [[V3]] : si32
    for (int i = 0; i < 100; ++i) {
        /* ... */
    }
}
