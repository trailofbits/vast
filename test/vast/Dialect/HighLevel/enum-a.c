// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

// CHECK: hl.enum @color : !hl.int< unsigned > {
// CHECK:  hl.enum.const @RED = #core.integer<0> : !hl.int
// CHECK:  hl.enum.const @GREEN = #core.integer<1> : !hl.int
// CHECK:  hl.enum.const @BLUE = #core.integer<2> : !hl.int
// CHECK: }
enum color { RED, GREEN, BLUE };

// CHECK: hl.var @c, <external> : !hl.lvalue<!hl.elaborated<!hl.enum<@color>>>
// CHECK: [[V1:%[0-9]+]] = hl.enumref @GREEN : !hl.int
// CHECK: [[V2:%[0-9]+]] = hl.implicit_cast [[V1]] IntegralCast : !hl.int -> !hl.elaborated<!hl.enum<@color>>
// CHECK: hl.value.yield [[V2]] : !hl.elaborated<!hl.enum<@color>>
enum color c = GREEN;

// CHECK: hl.typedef @color : !hl.elaborated<!hl.enum<@color>>
typedef enum color color;

// CHECK: hl.var @tc, <common> : !hl.lvalue<!hl.elaborated<!hl.typedef<@color>>>
color tc;
