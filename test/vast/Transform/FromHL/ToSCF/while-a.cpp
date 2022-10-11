// RUN: vast-cc --from-source %s | vast-opt --vast-hl-lower-types --vast-hl-structs-to-tuples --vast-hl-to-scf | FileCheck %s

void fn()
{
    // CHECK: [[V0:%[0-9]+]] = hl.var "x" : !hl.lvalue<ui1> = {
    // CHECK:   [[V11:%[0-9]+]] = hl.const #hl.bool<true> : ui1
    // CHECK:   hl.value.yield [[V11]] : ui1
    // CHECK: }
    // CHECK: scf.while : () -> () {
    // CHECK:   [[V21:%[0-9]+]] = hl.ref [[V0]] : !hl.lvalue<ui1>
    // CHECK:   [[V22:%[0-9]+]] = hl.implicit_cast [[V21]] LValueToRValue : !hl.lvalue<ui1> -> ui1
    // CHECK:   [[V23:%[0-9]+]] = hl.implicit_cast [[V22]] IntegralCast : ui1 -> i1
    // CHECK:   scf.condition([[V23]])
    // CHECK: } do {
    // CHECK:   [[V31:%[0-9]+]] = hl.ref [[V0]] : !hl.lvalue<ui1>
    // CHECK:   [[V32:%[0-9]+]] = hl.ref [[V0]] : !hl.lvalue<ui1>
    // CHECK:   [[V33:%[0-9]+]] = hl.implicit_cast [[V32]] LValueToRValue : !hl.lvalue<ui1> -> ui1
    // CHECK:   [[V34:%[0-9]+]] = hl.lnot [[V33]] : ui1
    // CHECK:   [[V35:%[0-9]+]] = hl.assign [[V34]] to [[V31]] : ui1, !hl.lvalue<ui1> -> ui1
    // CHECK:   scf.yield
    // CHECK: }
    // CHECK: hl.return
    bool x = true;
    while(x)
    {
        x = !x;
    }
}
