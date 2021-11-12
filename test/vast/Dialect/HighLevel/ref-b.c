// RUN: vast-cc --from-source %s | FileCheck %s
// RUN: vast-cc --from-source %s > %t && vast-opt %t | diff -B %t -

int main()
{
    int x = 0;

    // CHECK: [[V1:%[0-9]+]] = hl.declref @x : !hl.int
    // CHECK: [[V2:%[0-9]+]] = hl.addressof [[V1]] : !hl.int -> !hl.ptr<int>
    // CHECK: hl.var @y = [[V2]] : !hl.ptr<int>
    int *y = &x;

    // CHECK: [[V3:%[0-9]+]] = hl.declref @y : !hl.ptr<int>
    // CHECK: [[V4:%[0-9]+]] = hl.implicit_cast [[V3]] LValueToRValue : !hl.ptr<int> -> !hl.ptr<int>
    // CHECK: [[V5:%[0-9]+]] = hl.deref [[V4]] : !hl.ptr<int> -> !hl.int
    // CHECK: [[V6:%[0-9]+]] = hl.implicit_cast [[V5]] LValueToRValue : !hl.int -> !hl.int
    // CHECK: hl.var @z = [[V6]] : !hl.int
    int z = *y;
}
