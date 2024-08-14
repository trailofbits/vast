// RUN: %vast-front -vast-emit-mlir=hl %s -o - | %vast-opt --vast-hl-ude | %file-check %s

// CHECK-NOT: hl.struct "used"
struct unused_struct {};

// CHECK-NOT: hl.typedef @unused_typedef
typedef struct unused_struct unused_typedef;

// CHECK-NOT: hl.typedef @used_typedef
typedef unused_typedef unused_transitive_typedef;
