// RUN: vast-cc --ccopts -xc --from-source %s | FileCheck %s
// RUN: vast-cc --ccopts -xc --from-source %s > %t && vast-opt %t | diff -B %t -

// CHECK: hl.type.decl @Element
// CHECK: hl.typedef @Element : !hl.record<z : !hl.int, state : !hl.named_type<@State>>
struct Element {
    int z;
    enum State { SOLID, LIQUID, GAS, PLASMA } state;
// CHECK: hl.global @oxygen : !hl.named_type<@Element>
// CHECK:  [[V1:%[0-9]+]] = hl.constant.int 8 : !hl.int
// CHECK:  [[V2:%[0-9]+]] = hl.declref @GAS : !hl.int
// CHECK:  [[V3:%[0-9]+]] = hl.implicit_cast [[V2]] IntegralCast : !hl.int -> !hl.named_type<@State>
// CHECK:  hl.initlist [[V1]], [[V3]] : (!hl.int, !hl.named_type<@State>) -> !hl.named_type<@Element>
} oxygen = { 8, GAS };

void foo(void) {
    // CHECK: hl.var @e : !hl.named_type<@State>
    // CHECK:  hl.declref @LIQUID : !hl.int
    enum State e = LIQUID;
}
