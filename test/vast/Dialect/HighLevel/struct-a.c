// RUN: vast-cc --ccopts -xc --from-source %s | FileCheck %s
// RUN: vast-cc --ccopts -xc --from-source %s > %t && vast-opt %t | diff -B %t -

// CHECK: hl.struct "empty"
struct empty {};

// CHECK: hl.struct "pair" : {
// CHECK:  hl.field "a" : !hl.int
// CHECK:  hl.field "b" : !hl.int
// CHECK: }
struct pair {
  int a, b;
};

// CHECK: hl.var "p" : !hl.lvalue<!hl.elaborated<!hl.record<"pair">>>
struct pair p;

// CHECK: hl.type "forward"
struct forward;

// CHECK: hl.struct "forward" : {
// CHECK:  hl.field "a" : !hl.int
// CHECK: }
struct forward {
  int a;
};

// CHECK: hl.struct "wrap" : {
// CHECK:  hl.field "v" : !hl.int
// CHECK: }

// CHECK: hl.typedef "wrap_t" : !hl.elaborated<!hl.record<"wrap">>
typedef struct wrap {
  int v;
} wrap_t;

// CHECK: hl.var "w" : !hl.lvalue<!hl.elaborated<!hl.typedef<"wrap_t">>>
wrap_t w;

// CHECK: hl.struct "compound" : {
// CHECK:  hl.field "e" : !hl.elaborated<!hl.record<"empty">>
// CHECK:  hl.field "w" : !hl.elaborated<!hl.typedef<"wrap_t">>
// CHECK: }
struct compound {
  struct empty e;
  wrap_t w;
};

int main() {
  // CHECK: hl.var "e" : !hl.lvalue<!hl.elaborated<!hl.record<"empty">>>
  struct empty e;
}
