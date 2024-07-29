// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -
int foo();
int &bar();

int main() {
    int a;
    // CHECK: hl.binary_cond : !hl.int {
    // CHECK: hl.value.yield {{%[0-9]+}} : [[type:!hl.lvalue<!hl.int>]]
    // CHECK: }, {
    // CHECK: ^bb0([[arg:%[a-z0-9]+]]: [[type]])
    // CHECK: [[opaq:%[0-9]+]] = hl.opaque_expr [[arg]]
    // CHECK: hl.cond.yield [[opaq]] : !hl.int
    // CHECK: } ? {
    // CHECK: ^bb0([[arg:%[a-z0-9]+]]? [[type]])
    // CHECK: [[opaq:%[0-9]+]] = hl.opaque_expr [[arg]]
    // CHECK: hl.value.yield {{%[0-9]+}} : !hl.int
    // CHECK: } : {
    // CHECK: hl.value.yield {{%[0-9]+}} : !hl.int
    // CHECK: }
    int b = bar() ?: a;
}
