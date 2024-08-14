// RUN: %vast-front -vast-emit-mlir=hl %s -o - | %vast-opt --vast-hl-ude | %file-check %s

// CHECK-NOT: hl.typedecl "unused_struct"
struct unused_struct;

// CHECK-NOT: hl.typedef @unused_t
typedef struct unused_struct unused_t;

// CHECK-NOT: hl.var @unused
extern unused_t unused;
