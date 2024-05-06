// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

// CHECK: hl.struct "Element"
// CHECK:  hl.field "z" : !hl.int
// CHECK:  hl.enum "State" : !hl.int< unsigned >
// CHECK:   hl.enum.const "SOLID"
// CHECK:   hl.enum.const "LIQUID"
// CHECK:   hl.enum.const "GAS"
// CHECK:   hl.enum.const "PLASMA"
// CHECK:  hl.field "state" : !hl.elaborated<!hl.enum<"State">>
struct Element {
    int z;
    enum State { SOLID, LIQUID, GAS, PLASMA } state;

// CHECK: hl.var "oxygen" : !hl.lvalue<!hl.elaborated<!hl.record<"Element">>>
// CHECK:  [[V1:%[0-9]+]] = hl.const #core.integer<8> : !hl.int
// CHECK:  [[V2:%[0-9]+]] = hl.enumref "GAS" : !hl.int
// CHECK:  [[V3:%[0-9]+]] = hl.implicit_cast [[V2]] IntegralCast : !hl.int -> !hl.elaborated<!hl.enum<"State">>
// CHECK:  hl.initlist [[V1]], [[V3]] : (!hl.int, !hl.elaborated<!hl.enum<"State">>) -> !hl.elaborated<!hl.record<"Element">>
} oxygen = { 8, GAS };

void foo(void) {
    // CHECK: hl.var "e" : !hl.lvalue<!hl.elaborated<!hl.enum<"State">>>
    // CHECK:   hl.enumref "LIQUID" : !hl.int
    enum State e = LIQUID;
}
