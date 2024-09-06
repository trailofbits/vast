// RUN: %vast-front -vast-emit-mlir=hl %s -o - | %vast-opt --vast-hl-ude | %file-check %s

// CHECK-NOT: hl.struct @unused
struct unused {};

// CHECK-NOT: hl.typedef @unused_t
typedef struct unused unused_t;

// CHECK-NOT: hl.struct @unused_rty
struct unused_rty {};

// CHECK-NOT: hl.typedef @unused_rty_t
typedef struct unused_rty unused_rty_t;

// CHECK-NOT: hl.func @unused_declaration_with_typedefs
unused_rty_t unused_declaration_with_typedefs(unused_t);

// CHECK-NOT: hl.func @unused_declaration_with_structs
struct unused_rty unused_declaration_with_structs(struct unused);
