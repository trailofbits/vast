// RUN: vast-cc --ccopts -xc --from-source %s | FileCheck %s
// RUN: vast-cc --ccopts -xc --from-source %s > %t && vast-opt %t | diff -B %t -

// CHECK: hl.struct "s" : {
// CHECK:  hl.field "a" : !hl.int
// CHECK:  hl.field "b" : !hl.short
// CHECK: }
struct s {
    int a;
    short b;
};

// CHECK: func @f() -> !hl.void
void f() {
    // CHECK: [[V:%[0-9]+]] = hl.var "v" : !hl.lvalue<!hl.elaborated<!hl.record<"s">>>
    struct s v;
    // CHECK: hl.var "x" : !hl.lvalue<!hl.int>
    // CHECK:   [[V1:%[0-9]+]] = hl.ref [[V]] : !hl.lvalue<!hl.elaborated<!hl.record<"s">>>
    // CHECK:   [[V2:%[0-9]+]] = hl.member [[V1]] at "a" : !hl.lvalue<!hl.elaborated<!hl.record<"s">>> -> !hl.lvalue<!hl.int>
    // CHECK:   [[V3:%[0-9]+]] = hl.implicit_cast [[V2]] LValueToRValue : !hl.lvalue<!hl.int> -> !hl.int
    // CHECK:   hl.value.yield [[V3]] : !hl.int
    int x = v.a;
    // CHECK: [[V1:%[0-9]+]] = hl.ref [[V]] : !hl.lvalue<!hl.elaborated<!hl.record<"s">>>
    // CHECK: [[V2:%[0-9]+]] = hl.member [[V1]] at "b" : !hl.lvalue<!hl.elaborated<!hl.record<"s">>> -> !hl.lvalue<!hl.short>
    // CHECK: [[V3:%[0-9]+]] = hl.const #hl.integer<1> : !hl.int
    // CHECK: [[V4:%[0-9]+]] = hl.implicit_cast [[V3]] IntegralCast : !hl.int -> !hl.short
    // CHECK: hl.assign [[V4]] to [[V2]] : !hl.short
    v.b = 1;
}
