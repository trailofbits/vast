// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -
//
void foo() {
    // character literal in C matches int
    // CHECK: hl.generic_expr {selected = 1 : index} match region : {
    // CHECK: hl.type.yield {{%[0-9]+}} : !hl.int
    // CHECK: hl.generic_asoc match !hl.char
    // CHECK: hl.value.yield
    // CHECK: hl.generic_asoc match !hl.int
    // CHECK: hl.value.yield
    // CHECK: hl.generic_asoc : {
    // CHECK: hl.value.yield
    int x = _Generic('x', char: "2", int: 1, default: 3);
    // CHECK: hl.generic_expr {selected = 1 : index} match type : !hl.int {
    // CHECK: hl.generic_asoc match !hl.char
    // CHECK: hl.value.yield
    // CHECK: hl.generic_asoc match !hl.int
    // CHECK: hl.value.yield
    // CHECK: hl.generic_asoc : {
    // CHECK: hl.value.yield
    x = _Generic(int, char: "2", int: 1, default: 3);
    // CHECK: hl.generic_expr {selected = 2 : index} match type : !hl.float {
    // CHECK: hl.generic_asoc match !hl.char
    // CHECK: hl.value.yield
    // CHECK: hl.generic_asoc match !hl.int
    // CHECK: hl.value.yield
    // CHECK: hl.generic_asoc : {
    // CHECK: hl.value.yield
    x = _Generic(float, char: "2", int: 1, default: 3);
    // CHECK: hl.generic_expr {selected = 1 : index} match type : !hl.float {
    // CHECK: hl.generic_asoc match !hl.int
    // CHECK: hl.value.yield
    // CHECK: hl.generic_asoc : {
    // CHECK: hl.value.yield
    // CHECK: hl.generic_asoc match !hl.char
    // CHECK: hl.value.yield
    x = _Generic(float, int: 1, default: 3, char: "2");
}
