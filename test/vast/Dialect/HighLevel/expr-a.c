// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

// CHECK: assign_to_lvalue_expr {{.*}} ([[A:%arg[0-9]+]]: !hl.lvalue<!hl.ptr<!hl.int>>)
void assign_to_lvalue_expr(int *a) {
    // CHECK: hl.expr : !hl.lvalue<!hl.int>
    (*a) = 1;
}
