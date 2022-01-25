// RUN: vast-cc --ccopts -xc --from-source %s | FileCheck %s
// RUN: vast-cc --ccopts -xc --from-source %s > %t && vast-opt %t | diff -B %t -

// CHECK: hl.type.decl @empty
// CHECK: hl.typedef @empty : !hl.record<>
struct empty {};

// CHECK: hl.type.decl @pair
// CHECK: hl.typedef @pair : !hl.record<a : !hl.int, b : !hl.int>
struct pair {
  int a, b;
};

// CHECK: hl.global @p : !hl.named_type<@pair>
struct pair p;

// CHECK: hl.type.decl @forward
struct forward;

// CHECK: hl.typedef @forward : !hl.record<a : !hl.int>
struct forward {
  int a;
};

// CHECK: hl.type.decl @wrap
// CHECK: hl.typedef @wrap : !hl.record<v : !hl.int>
typedef struct wrap {
  int v;
} wrap;

// CHECK: hl.global @w : !hl.named_type<@wrap>
wrap w;

// CHECK: hl.type.decl @nested
// CHECK: hl.typedef @nested : !hl.record<>
struct nested {};

// CHECK: hl.type.decl @compound
// CHECK: hl.typedef @compound : !hl.record<n : !hl.named_type<@nested>, w : !hl.named_type<@wrap>>
struct compound {
  struct nested n;
  wrap w;
};

int main() {
  // CHECK: hl.var @e : !hl.named_type<@empty>
  struct empty e;
}
