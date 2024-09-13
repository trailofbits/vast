// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o - | %file-check %s
// RUN: %vast-cc1 -vast-emit-mlir=hl %s -o %t && %vast-opt %t | diff -B %t -

int main()
{
    // CHECK: hl.var @x : !hl.lvalue<!hl.int>
    int x = 0;

    // CHECK: hl.var @y : !hl.lvalue<!hl.ptr<!hl.int>>
    // CHECK:   [[V1:%[0-9]+]] = hl.ref @x
    // CHECK:   [[V2:%[0-9]+]] = hl.addressof [[V1]] : !hl.lvalue<!hl.int> -> !hl.ptr<!hl.int>
    int *y = &x;

    // CHECK: hl.var @z : !hl.lvalue<!hl.int>
    // CHECK: [[V3:%[0-9]+]] = hl.ref @y
    // CHECK: [[V4:%[0-9]+]] = hl.implicit_cast [[V3]] LValueToRValue : !hl.lvalue<!hl.ptr<!hl.int>> -> !hl.ptr<!hl.int>
    // CHECK: [[V5:%[0-9]+]] = hl.deref [[V4]] : !hl.ptr<!hl.int> -> !hl.lvalue<!hl.int>
    // CHECK: [[V6:%[0-9]+]] = hl.implicit_cast [[V5]] LValueToRValue : !hl.lvalue<!hl.int> -> !hl.int
    int z = *y;
}
