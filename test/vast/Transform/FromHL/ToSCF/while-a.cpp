// RUN: vast-cc --from-source %s | vast-opt --vast-hl-lower-types --vast-hl-structs-to-tuples --vast-hl-lower-cf | FileCheck %s

void fn()
{
    // CHECK: [[V0:%[0-9]+]] = hl.var "x" : !hl.lvalue<i1> = {
    // CHECK:   [[V11:%[0-9]+]] = hl.constant.int true : i1
    // CHECK:   hl.value.yield [[V11]] : i1
    // CHECK: }
    // CHECK: scf.while : () -> () {
    // CHECK:   [[V21:%[0-9]+]] = hl.decl.ref [[V0]] : !hl.lvalue<i1>
    // CHECK:   [[V22:%[0-9]+]] = hl.implicit_cast [[V21]] LValueToRValue : !hl.lvalue<i1> -> i1
    // CHECK:   scf.condition([[V22]])
    // CHECK: } do {
    // CHECK:   [[V31:%[0-9]+]] = hl.decl.ref [[V0]] : !hl.lvalue<i1>
    // CHECK:   [[V32:%[0-9]+]] = hl.decl.ref [[V0]] : !hl.lvalue<i1>
    // CHECK:   [[V33:%[0-9]+]] = hl.implicit_cast [[V32]] LValueToRValue : !hl.lvalue<i1> -> i1
    // CHECK:   [[V34:%[0-9]+]] = hl.lnot [[V33]] : i1
    // CHECK:   [[V35:%[0-9]+]] = hl.assign [[V34]] to [[V31]] : i1
    // CHECK:   scf.yield
    // CHECK: }
    // CHECK: hl.return
    bool x = true;
    while(x)
    {
        x = !x;
    }
}
