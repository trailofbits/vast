// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

// CHECK: hl.enum "color" : !hl.int< unsigned >
enum color { RED, GREEN, BLUE };
// CHECK: hl.var "r" : !hl.lvalue<!hl.elaborated<!hl.enum<"color">>>
// CHECK:  hl.enumref "RED" : !hl.int
enum color r = RED;

// CHECK: hl.typedef "color_t" : !hl.elaborated<!hl.enum<"color">>
// CHECK: hl.var "x" : !hl.lvalue<!hl.elaborated<!hl.typedef<"color_t">>>
// CHECK:  hl.enumref "GREEN" : !hl.int
typedef enum color color_t;
color_t x = GREEN;
