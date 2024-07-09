// RUN: %vast-front -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-front -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

#include <stddef.h>

struct S {
  float f;
  double d[10];
};

struct T {
  int i;
  struct S s[10];
};

int main() {
    // CHECK: hl.offsetof.expr [#hl.offset_of_node<identifier : "T::s">, #hl.offset_of_node<index : 0>, #hl.offset_of_node<identifier : "S::d">, #hl.offset_of_node<index : 1>] {{.*}} {
    // CHECK: hl.value.yield
    // CHECK: }, {
    // CHECK: hl.value.yield
    // CHECK: }
    return offsetof(struct T, s[2].d[3]);
}
