// RUN: vast-cc --ccopts -xc --from-source %s | FileCheck %s
// RUN: vast-cc --ccopts -xc --from-source %s > %t && vast-opt %t | diff -B %t -

// CHECK: hl.type.decl @struct.empty
// CHECK: hl.record @struct.empty
struct empty {};

// CHECK: hl.type.decl @struct.pair
// CHECK: hl.record @struct.pair : {
// CHECK:  hl.field @a : !hl.int
// CHECK:  hl.field @b : !hl.int
// CHECK: }
struct pair {
  int a, b;
};

// CHECK: hl.var @p : !hl.named_type<@struct.pair>
struct pair p;

// CHECK: hl.type.decl @struct.forward
struct forward;

// CHECK: hl.record @struct.forward : {
// CHECK:  hl.field @a : !hl.int
// CHECK: }
struct forward {
  int a;
};

// CHECK: hl.type.decl @struct.wrap
// CHECK: hl.record @struct.wrap : {
// CHECK:  hl.field @v : !hl.int
// CHECK: }

// CHECK: hl.typedef @wrap_t : !hl.named_type<@struct.wrap>
typedef struct wrap {
  int v;
} wrap_t;

// CHECK: hl.var @w : !hl.named_type<@wrap_t>
wrap_t w;

// CHECK: hl.type.decl @struct.compound
// CHECK: hl.record @struct.compound : {
// CHECK:  hl.field @e : !hl.named_type<@struct.empty>
// CHECK:  hl.field @w : !hl.named_type<@wrap_t>
// CHECK: }
struct compound {
  struct empty e;
  wrap_t w;
};

int main() {
  // CHECK: hl.var @e : !hl.named_type<@struct.empty>
  struct empty e;
}
