// RUN: vast-cc --ccopts -xc --from-source %s | vast-opt --vast-hl-lower-types --vast-hl-structs-to-tuples --vast-hl-to-scf | FileCheck %s

// CHECK:   func @fn(%arg0: !hl.lvalue<si32>) -> none {
void fn(int x)
{
    // CHECK: scf.while : () -> () {
    // CHECK:   [[V0:%[0-9]+]] = hl.ref %arg0 : !hl.lvalue<si32>
    // CHECK:   [[V1:%[0-9]+]] = hl.implicit_cast [[V0]] LValueToRValue : !hl.lvalue<si32> -> si32
    // CHECK:   [[V2:%[0-9]+]] = hl.const #hl.integer<0> : si32
    // CHECK:   [[V3:%[0-9]+]] = hl.cmp ne [[V1]], [[V2]] : si32, si32 -> si32
    // CHECK:   [[V4:%[0-9]+]] = hl.implicit_cast [[V3]] IntegralCast : si32 -> i1
    // CHECK:   scf.condition([[V4]])
    // CHECK: } do {
    // CHECK:   [[V10:%[0-9]+]] = hl.ref %arg0 : !hl.lvalue<si32>
    // CHECK:   [[V11:%[0-9]+]] = hl.pre.dec [[V10]] : !hl.lvalue<si32>
    // CHECK:   scf.yield
    // CHECK: }
    // CHECK: hl.return
    while (x != 0)
        --x;
}
