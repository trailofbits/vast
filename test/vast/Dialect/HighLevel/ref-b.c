// RUN: vast-cc --from-source %s | FileCheck %s
// RUN: vast-cc --from-source %s > %t && vast-opt %t | diff -B %t -

int main()
{
    // CHECK: [[X:%[0-9]+]] = hl.var "x" : !hl.lvalue<!hl.int>
    int x = 0;

    // CHECK: [[Y:%[0-9]+]] = hl.var "y" : !hl.lvalue<!hl.ptr<!hl.int>>
    // CHECK:   [[V1:%[0-9]+]] = hl.decl.ref [[X]] : !hl.lvalue<!hl.int>
    // CHECK:   [[V2:%[0-9]+]] = hl.addressof [[V1]] : !hl.lvalue<!hl.int> -> !hl.ptr<!hl.int>
    int *y = &x;

    // CHECK: [[Z:%[0-9]+]] = hl.var "z" : !hl.lvalue<!hl.int>
    // CHECK: [[V3:%[0-9]+]] = hl.decl.ref [[Y]] : !hl.lvalue<!hl.ptr<!hl.int>>
    // CHECK: [[V4:%[0-9]+]] = hl.implicit_cast [[V3]] LValueToRValue : !hl.lvalue<!hl.ptr<!hl.int>> -> !hl.ptr<!hl.int>
    // CHECK: [[V5:%[0-9]+]] = hl.deref [[V4]] : !hl.ptr<!hl.int> -> !hl.lvalue<!hl.int>
    // CHECK: [[V6:%[0-9]+]] = hl.implicit_cast [[V5]] LValueToRValue : !hl.lvalue<!hl.int> -> !hl.int
    int z = *y;
}
