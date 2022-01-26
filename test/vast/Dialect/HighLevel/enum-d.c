// RUN: vast-cc --ccopts -xc --from-source %s | FileCheck %s
// RUN: vast-cc --ccopts -xc --from-source %s > %t && vast-opt %t | diff -B %t -

// CHECK: hl.enum.decl @color : !hl.int<unsigned>
enum color { RED, GREEN, BLUE };
// CHECK: hl.global @r : !hl.named_type<@color>
// CHECK:  hl.declref @RED : !hl.int
enum color r = RED;

// CHECK: hl.type.decl @color_t
// CHECK: hl.typedef @color_t : !hl.named_type<@color>
// CHECK: hl.global @x : !hl.named_type<@color>
// CHECK:  hl.declref @GREEN : !hl.int
typedef enum color color_t;
color_t x = GREEN;
