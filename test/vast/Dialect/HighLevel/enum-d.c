// RUN: vast-cc --ccopts -xc --from-source %s | FileCheck %s
// RUN: vast-cc --ccopts -xc --from-source %s > %t && vast-opt %t | diff -B %t -

// CHECK: hl.enum.decl @enum.color : !hl.int<unsigned>
enum color { RED, GREEN, BLUE };
// CHECK: hl.global @r : !hl.named_type<@enum.color>
// CHECK:  hl.declref @RED : !hl.int
enum color r = RED;

// CHECK: hl.typedef @color_t : !hl.named_type<@enum.color>
// CHECK: hl.global @x : !hl.named_type<@color_t>
// CHECK:  hl.declref @GREEN : !hl.int
typedef enum color color_t;
color_t x = GREEN;
