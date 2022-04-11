// RUN: vast-cc --ccopts -xc --from-source %s | FileCheck %s
// RUN: vast-cc --ccopts -xc --from-source %s > %t && vast-opt %t | diff -B %t -

// CHECK: hl.record "struct Element"
// CHECK:  hl.field "z" : !hl.int
// CHECK:  hl.enum.decl "enum State" : !hl.int<unsigned>
// CHECK:   hl.enum.const "SOLID"
// CHECK:   hl.enum.const "LIQUID"
// CHECK:   hl.enum.const "GAS"
// CHECK:   hl.enum.const "PLASMA"
// CHECK:  hl.field "state" : !hl.named_type<"enum State">
struct Element {
    int z;
    enum State { SOLID, LIQUID, GAS, PLASMA } state;

// CHECK: hl.var "oxygen" : !hl.lvalue<!hl.named_type<"struct Element">>
// CHECK:  [[V1:%[0-9]+]] = hl.constant.int 8 : !hl.int
// CHECK:  [[V2:%[0-9]+]] = hl.enumref "GAS" : !hl.int
// CHECK:  [[V3:%[0-9]+]] = hl.implicit_cast [[V2]] IntegralCast : !hl.int -> !hl.named_type<"enum State">
// CHECK:  hl.initlist [[V1]], [[V3]] : (!hl.int, !hl.named_type<"enum State">) -> !hl.named_type<"struct Element">
} oxygen = { 8, GAS };

void foo(void) {
    // CHECK: hl.var "e" : !hl.lvalue<!hl.named_type<"enum State">>
    // CHECK:   hl.enumref "LIQUID" : !hl.int
    enum State e = LIQUID;
}
