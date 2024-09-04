// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

void fn() {
    int x, y;
    // CHECK: hl.choose_expr cond false : !hl.lvalue<!hl.int> {
    // CHECK: hl.cond.yield {{%.*}} : !hl.int
    // CHECK: hl.value.yield {{%.*}} : !hl.lvalue<!hl.int>
    // CHECK: hl.value.yield {{%.*}} : !hl.lvalue<!hl.int>
    int z = __builtin_choose_expr(0, x, y);
    // CHECK: hl.choose_expr cond true : !hl.lvalue<!hl.int> {
    int w = __builtin_choose_expr(1, x, y);
}
