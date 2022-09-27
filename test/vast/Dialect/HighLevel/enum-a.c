// RUN: vast-cc --ccopts -xc --from-source %s | FileCheck %s
// RUN: vast-cc --ccopts -xc --from-source %s > %t && vast-opt %t | diff -B %t -

// CHECK: hl.enum "color" : !hl.int< unsigned > {
// CHECK:  hl.enum.const "RED" = #hl.integer<0> : !hl.int
// CHECK:  hl.enum.const "GREEN" = #hl.integer<1> : !hl.int
// CHECK:  hl.enum.const "BLUE" = #hl.integer<2> : !hl.int
// CHECK: }
enum color { RED, GREEN, BLUE };

// CHECK: hl.var "c" : !hl.lvalue<!hl.elaborated<!hl.record<"color">>>
// CHECK: [[V1:%[0-9]+]] = hl.enumref "GREEN" : !hl.int
// CHECK: [[V2:%[0-9]+]] = hl.implicit_cast [[V1]] IntegralCast : !hl.int -> !hl.elaborated<!hl.record<"color">>
// CHECK: hl.value.yield [[V2]] : !hl.elaborated<!hl.record<"color">>
enum color c = GREEN;

// CHECK: hl.typedef "color" : !hl.elaborated<!hl.record<"color">>
typedef enum color color;

// CHECK: hl.var "tc" : !hl.lvalue<!hl.typedef<"color">>
color tc;
