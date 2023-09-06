// RUN: %vast-cc --ccopts -xc --from-source %s | FileCheck %s
// RUN: %vast-cc --ccopts -xc --from-source %s > %t && %vast-opt %t | diff -B %t -

// CHECK: hl.enum "color" : !hl.int< unsigned >
enum color { RED, GREEN, BLUE };
// CHECK: hl.var "r" : !hl.lvalue<!hl.elaborated<!hl.record<"color">>>
// CHECK:  hl.enumref "RED" : !hl.int
enum color r = RED;

// CHECK: hl.typedef "color_t" : !hl.elaborated<!hl.record<"color">>
// CHECK: hl.var "x" : !hl.lvalue<!hl.elaborated<!hl.typedef<"color_t">>>
// CHECK:  hl.enumref "GREEN" : !hl.int
typedef enum color color_t;
color_t x = GREEN;
