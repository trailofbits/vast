// RUN: vast-cc --from-source %s | FileCheck %s
// RUN: vast-cc --from-source %s > %t && vast-opt %t | diff -B %t -

int main()
{
    int x = 0;

    // CHECK: hl.var @y : !hl.ptr<!hl.int>
    // CHECK: [[V1:%[0-9]+]] = hl.declref @x : !hl.int
    // CHECK: [[V2:%[0-9]+]] = hl.addressof [[V1]] : !hl.int -> !hl.ptr<!hl.int>
    int *y = &x;

    // CHECK: hl.var @z : !hl.int
    // CHECK: [[V3:%[0-9]+]] = hl.declref @y : !hl.ptr<!hl.int>
    // CHECK: [[V4:%[0-9]+]] = hl.implicit_cast [[V3]] LValueToRValue : !hl.ptr<!hl.int> -> !hl.ptr<!hl.int>
    // CHECK: [[V5:%[0-9]+]] = hl.deref [[V4]] : !hl.ptr<!hl.int> -> !hl.int
    // CHECK: [[V6:%[0-9]+]] = hl.implicit_cast [[V5]] LValueToRValue : !hl.int -> !hl.int
    int z = *y;
}
